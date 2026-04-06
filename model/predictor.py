# -*- coding: utf-8 -*-
"""
模型推理模块
集成推理（EMA + SWA + TopK）和 Transformer 因子序列计算
从 TransformerStock.py 的 predict_stocks() 和 calculate_transformer_factor_series() 提取
"""
import os
import time
import logging
from typing import Optional, List, Dict

from data.indicators_no_transformer import safe_sma
from utils.stock_filter import should_intercept_stock

import numpy as np
import pandas as pd
import torch
import joblib
import talib as ta
from tqdm import tqdm

from config import get_settings
from data.types import FEATURES
from model.transformer import StockTransformer
from exceptions import ModelLoadError

logger = logging.getLogger(__name__)

# 延迟导入 efinance
try:
    import efinance as ef
    _EF_AVAILABLE = True
except ImportError:
    _EF_AVAILABLE = False
    ef = None


def _load_ensemble_models(device: torch.device) -> List[tuple]:
    """动态加载所有可用模型（EMA + SWA + TopK）"""
    settings = get_settings()
    models = []

    # EMA 模型
    if os.path.exists(settings.paths.model_path):
        try:
            m = StockTransformer(input_dim=len(FEATURES), lookback_days=settings.model.lookback_days).to(device)
            m.load_state_dict(torch.load(settings.paths.model_path, map_location=device))
            m.eval()
            models.append(("EMA", m))
        except Exception as e:
            raise ModelLoadError(f"EMA模型加载失败: {e}", model_type="EMA", path=settings.paths.model_path)

    # SWA 模型
    if os.path.exists(settings.paths.swa_model_path):
        try:
            m = StockTransformer(input_dim=len(FEATURES), lookback_days=settings.model.lookback_days).to(device)
            swa_state = torch.load(settings.paths.swa_model_path, map_location=device)
            clean_state = {k.replace('module.', ''): v for k, v in swa_state.items() if k != 'n_averaged'}
            m.load_state_dict(clean_state)
            m.eval()
            models.append(("SWA", m))
        except Exception as e:
            logger.warning(f"SWA 模型加载失败: {e}")

    # Top-K 模型
    topk_dir = settings.paths.topk_checkpoint_dir
    if os.path.exists(topk_dir):
        for fname in os.listdir(topk_dir):
            if fname.endswith(".pth"):
                try:
                    m = StockTransformer(input_dim=len(FEATURES), lookback_days=settings.model.lookback_days).to(device)
                    ckpt = torch.load(os.path.join(topk_dir, fname), map_location=device)
                    clean_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items() if k != 'n_averaged'}
                    m.load_state_dict(clean_ckpt)
                    m.eval()
                    models.append((f"TopK_{fname}", m))
                except Exception as e:
                    logger.warning(f"TopK 模型 {fname} 加载失败: {e}")

    return models


def _load_scalers() -> tuple:
    """加载 scaler（独立 + 全局）"""
    settings = get_settings()
    scalers = {}
    global_scaler = None

    if os.path.exists(settings.paths.scaler_path):
        scalers = joblib.load(settings.paths.scaler_path)
    if os.path.exists(settings.paths.global_scaler_path):
        global_scaler = joblib.load(settings.paths.global_scaler_path)

    return scalers, global_scaler


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """从原始 OHLCV 数据计算所有特征"""
    temp = df.copy()
    temp['MA5'] = safe_sma(temp['Close'], period=5)
    temp['MA10'] = safe_sma(temp['Close'], period=10)
    temp['MA20'] = safe_sma(temp['Close'], period=20)
    temp['MACD'], temp['MACD_Signal'], temp['MACD_Hist'] = ta.MACD(temp['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    temp['K'], temp['D'] = ta.STOCH(temp['High'], temp['Low'], temp['Close'], fastk_period=9, slowk_period=3, slowd_period=3)
    temp['J'] = 3 * temp['K'] - 2 * temp['D']
    temp['RSI'] = ta.RSI(temp['Close'], timeperiod=14)
    temp['ADX'] = ta.ADX(temp['High'], temp['Low'], temp['Close'], timeperiod=14)
    temp['BB_Upper'], temp['BB_Middle'], temp['BB_Lower'] = ta.BBANDS(temp['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    temp['OBV'] = ta.OBV(temp['Close'], temp['Volume'])
    temp['CCI'] = ta.CCI(temp['High'], temp['Low'], temp['Close'], timeperiod=20)
    temp.bfill(inplace=True)
    temp.dropna(inplace=True)
    return temp


def _select_scaler(code: str, scalers: dict, global_scaler, features_data: np.ndarray):
    """选择合适的 scaler 并标准化"""
    if code in scalers:
        return scalers[code].transform(features_data), "专用"
    elif global_scaler is not None:
        return global_scaler.transform(features_data), "全局"
    else:
        median = np.median(features_data, axis=0)
        q75 = np.percentile(features_data, 75, axis=0)
        q25 = np.percentile(features_data, 25, axis=0)
        iqr = np.clip(q75 - q25, 1e-6, None)
        return (features_data - median) / iqr, "在线标准化"


def predict_stocks(target_codes: List[str], models: Optional[List] = None) -> pd.DataFrame:
    """集成预测多只股票

    Args:
        target_codes: 股票代码列表
        models: 预加载的模型列表（可选，为None则自动加载）

    Returns:
        预测结果 DataFrame，按 expected_score 降序排列
    """

    settings = get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if models is None:
        models = _load_ensemble_models(device)

    if not models:
        logger.error("没有找到任何模型文件！")
        return pd.DataFrame()

    scalers, global_scaler = _load_scalers()
    predictions = []

    for i, target_code in enumerate(target_codes):
        try:
            time.sleep(1)
            if i > 0 and i % 100 == 0:
                logger.warning('每100只暂停100秒')
                time.sleep(100)

            if not _EF_AVAILABLE:
                continue

            df = ef.stock.get_quote_history(target_code, beg='20210101', end='20270101')
            if df is None or df.empty:
                continue
            if '日期' not in df.columns:
                continue

            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期').sort_index()

            temp = pd.DataFrame(index=df.index)
            temp['Open'] = df['开盘']
            temp['High'] = df['最高']
            temp['Low'] = df['最低']
            temp['Close'] = df['收盘']
            temp['Volume'] = df['成交量']
            temp['Turnover Rate'] = df['换手率']

            temp = _prepare_features(temp)
            # >>> 新增：ST/退市/停牌 统一拦截
            # 注意：此处在线拉取无法获取股票名称，传空字符串主要拦截停牌和空数据
            skip, reason = should_intercept_stock(target_code, "", temp)
            if skip:
                logger.warning(f"[拦截-预测] 跳过 {target_code}: {reason}")
                continue
            # <<< 新增结束
            if len(temp) < settings.model.lookback_days:
                continue

            scaled, scaler_type = _select_scaler(
                target_code, scalers, global_scaler, temp[FEATURES].values
            )

            last_seq = torch.FloatTensor(scaled[-settings.model.lookback_days:]).unsqueeze(0).to(device)

            # 集成推理
            all_probs = []
            all_rets = []
            with torch.no_grad():
                for name, m in models:
                    mean_probs, _, mean_ret = m.mc_predict(last_seq, n_forward=10)
                    all_probs.append(mean_probs)
                    all_rets.append(mean_ret)

            ensemble_probs = torch.stack(all_probs).mean(dim=0)
            ensemble_ret = torch.stack(all_rets).mean(dim=0)
            inter_model_uncertainty = torch.stack(all_probs).std(dim=0).mean().item()

            probs_1d = ensemble_probs.squeeze(0)
            up_prob = (probs_1d[2] + probs_1d[3]).item()
            pred_ret = ensemble_ret.item()
            trend = "上涨" if up_prob > 0.5 else "下跌"
            confidence = up_prob if trend == "上涨" else (1 - up_prob)
            risk_flag = "⚠️ 高模型分歧" if inter_model_uncertainty > 0.05 else "正常"
            expected_score = up_prob * pred_ret if pred_ret > 0 else pred_ret * (1 - up_prob)

            predictions.append({
                'code': target_code,
                'trend': trend,
                'probability': round(float(confidence), 4),
                'predicted_ret': round(float(pred_ret), 4),
                'uncertainty': round(float(inter_model_uncertainty), 4),
                'risk_flag': risk_flag,
                'expected_score': round(float(expected_score), 4),
                'ensemble_size': len(models),
                'scaler_type': scaler_type,
            })

        except Exception as e:
            logger.warning(f"{target_code} 预测失败: {e}")

    if predictions:
        df_result = pd.DataFrame(predictions)
        df_result = df_result.sort_values(by='expected_score', ascending=False).reset_index(drop=True)
        df_result.insert(0, 'rank', df_result.index + 1)
        return df_result
    return pd.DataFrame()


def calculate_transformer_factor_series(
    df: pd.DataFrame,
    code: str,
    device=None,
    lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    """为回测计算 Transformer 因子的历史序列

    自动加载集成模型进行批量滑动窗口推理

    Args:
        df: 个股全量数据，index 为日期
        code: 股票代码
        device: 推理设备
        lookback_days: 回看天数

    Returns:
        DataFrame with columns: transformer_prob, transformer_pred_ret, transformer_uncertainty
    """
    settings = get_settings()
    if lookback_days is None:
        lookback_days = settings.model.lookback_days
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        models = _load_ensemble_models(device)
        if not models:
            logger.warning(f"找不到任何模型，跳过 {code}")
            return pd.DataFrame(
                index=df.index,
                columns=['transformer_prob', 'transformer_pred_ret', 'transformer_uncertainty'],
            ).fillna(0.5)

        temp = df.copy()
        col_mapping = {'开盘': 'Open', '最高': 'High', '最低': 'Low', '收盘': 'Close', '成交量': 'Volume', '换手率': 'Turnover Rate'}
        for cn, en in col_mapping.items():
            if cn in temp.columns and en not in temp.columns:
                temp[en] = temp[cn]

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if any(c not in temp.columns for c in required):
            return pd.DataFrame(
                index=df.index,
                columns=['transformer_prob', 'transformer_pred_ret', 'transformer_uncertainty'],
            ).fillna(0.5)

        temp['Turnover Rate'] = temp.get('Turnover Rate', pd.Series(3.0, index=temp.index)).fillna(3.0)
        temp = _prepare_features(temp)

        if len(temp) < lookback_days:
            return pd.DataFrame(
                index=df.index,
                columns=['transformer_prob', 'transformer_pred_ret', 'transformer_uncertainty'],
            ).fillna(0.5)

        scalers, global_scaler = _load_scalers()
        scaled_data, _ = _select_scaler(code, scalers, global_scaler, temp[FEATURES].values)

        sequences = [scaled_data[i - lookback_days: i] for i in range(lookback_days, len(scaled_data) + 1)]
        valid_indices = [temp.index[i - 1] for i in range(lookback_days, len(scaled_data) + 1)]

        if not sequences:
            return pd.DataFrame(
                index=df.index,
                columns=['transformer_prob', 'transformer_pred_ret', 'transformer_uncertainty'],
            ).fillna(0.5)

        # 集成批量推理
        all_probs_list = []
        all_rets_list = []

        for name, m in models:
            probs_per_model = []
            rets_per_model = []
            m.eval()
            for seq in tqdm(sequences, desc=f"集成回测 {code} ({len(models)}模)", leave=False):
                with torch.no_grad():
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                    mean_probs, _, mean_ret = m.mc_predict(seq_tensor, n_forward=3)
                    probs_per_model.append(mean_probs.squeeze(0).cpu())
                    rets_per_model.append(mean_ret.item())
            all_probs_list.append(torch.stack(probs_per_model))
            all_rets_list.append(rets_per_model)

        stack_probs = torch.stack(all_probs_list)
        ensemble_probs = stack_probs.mean(dim=0)
        inter_model_std = stack_probs.std(dim=0).mean(dim=-1).numpy()
        up_probs = (ensemble_probs[:, 2] + ensemble_probs[:, 3]).numpy()
        mean_rets = np.array(all_rets_list).mean(axis=0)

        result_df = pd.DataFrame({
            'transformer_prob': up_probs,
            'transformer_pred_ret': mean_rets,
            'transformer_uncertainty': inter_model_std,
        }, index=valid_indices).reindex(df.index)

        result_df['transformer_prob'] = result_df['transformer_prob'].fillna(0.5)
        result_df['transformer_pred_ret'] = result_df['transformer_pred_ret'].fillna(0.0)
        result_df['transformer_uncertainty'] = result_df['transformer_uncertainty'].fillna(0.15)

        return result_df

    except Exception as e:
        logger.error(f"Transformer 因子计算错误 ({code}): {e}")
        return pd.DataFrame(
            index=df.index,
            columns=['transformer_prob', 'transformer_pred_ret', 'transformer_uncertainty'],
        ).fillna(0.5)

# -*- coding: utf-8 -*-
"""
独立预测入口脚本（无Transformer版本）
仅使用传统技术因子进行预测，不依赖 Transformer 模型
结果保存到独立目录，不覆盖原有Transformer版本的结果
"""
import os
import json
import logging
import argparse
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import talib as ta
from tqdm import tqdm

from config import get_settings, STOCK_CODES
from data.indicators_no_transformer import safe_sma
from data.types import NON_FACTOR_COLS
from utils.stock_filter import filter_codes_by_name, should_intercept_stock

# 延迟导入 efinance
try:
    import efinance as ef

    _EF_AVAILABLE = True
except ImportError:
    _EF_AVAILABLE = False
    ef = None

logger = logging.getLogger(__name__)


def calculate_traditional_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算传统技术因子（不使用Transformer模型）

    Args:
        df: 包含 OHLCV 数据的 DataFrame

    Returns:
        添加了技术因子的 DataFrame
    """
    temp = df.copy()

    # 均线
    temp['MA5'] = safe_sma(temp['Close'], period=5)
    temp['MA10'] = safe_sma(temp['Close'], period=10)
    temp['MA20'] = safe_sma(temp['Close'], period=20)

    # MACD
    temp['MACD'], temp['MACD_Signal'], temp['MACD_Hist'] = ta.MACD(
        temp['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )

    # KDJ
    temp['K'], temp['D'] = ta.STOCH(
        temp['High'], temp['Low'], temp['Close'],
        fastk_period=9, slowk_period=3, slowd_period=3
    )
    temp['J'] = 3 * temp['K'] - 2 * temp['D']

    # RSI
    temp['RSI'] = ta.RSI(temp['Close'], timeperiod=14)

    # ADX
    temp['ADX'] = ta.ADX(temp['High'], temp['Low'], temp['Close'], timeperiod=14)

    # 布林带
    temp['BB_Upper'], temp['BB_Middle'], temp['BB_Lower'] = ta.BBANDS(
        temp['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )

    # OBV
    temp['OBV'] = ta.OBV(temp['Close'], temp['Volume'])

    # CCI
    temp['CCI'] = ta.CCI(temp['High'], temp['Low'], temp['Close'], timeperiod=20)

    # ATR
    temp['ATR'] = ta.ATR(temp['High'], temp['Low'], temp['Close'], timeperiod=14)

    # 填充缺失值
    temp.bfill(inplace=True)
    temp.dropna(inplace=True)

    return temp


def calculate_factor_score(df: pd.DataFrame) -> float:
    """计算综合因子得分（仅使用传统因子）

    基于多个技术指标的综合评估：
    1. 趋势因子：MA5/MA10/MA20 多头排列
    2. 动量因子：MACD、RSI、KDJ
    3. 波动因子：布林带位置、ATR
    4. 成交量因子：OBV 趋势

    Returns:
        综合得分 (0-1)
    """
    if df is None or len(df) < 30:
        return 0.5

    last = df.iloc[-1]
    prev = df.iloc[-2]

    scores = []

    weights = []

    # 1. 均线趋势得分 (权重: 0.25)
    ma_score = 0.5
    if last['Close'] > last['MA5'] > last['MA10'] > last['MA20']:
        ma_score = 0.9  # 完美多头排列
    elif last['Close'] > last['MA5'] and last['MA5'] > last['MA10']:
        ma_score = 0.7  # 短期多头
    elif last['Close'] < last['MA5'] < last['MA10'] < last['MA20']:
        ma_score = 0.1  # 空头排列
    elif last['Close'] < last['MA5'] and last['MA5'] < last['MA10']:
        ma_score = 0.3  # 短期空头
    scores.append(ma_score)
    weights.append(0.25)

    # 2. MACD 得分 (权重: 0.20)
    macd_score = 0.5
    if last['MACD'] > 0 and last['MACD_Hist'] > 0:
        macd_score = 0.8  # 金叉且在零轴上方
    elif last['MACD'] > 0 and last['MACD_Hist'] > prev['MACD_Hist']:
        macd_score = 0.7  # 零轴上方，柱子增长
    elif last['MACD'] < 0 and last['MACD_Hist'] > 0:
        macd_score = 0.6  # 金叉形成中
    elif last['MACD'] < 0 and last['MACD_Hist'] < prev['MACD_Hist']:
        macd_score = 0.3  # 死叉且柱子缩短
    scores.append(macd_score)
    weights.append(0.20)

    # 3. RSI 得分 (权重: 0.15)
    rsi_score = 0.5
    if 40 <= last['RSI'] <= 60:
        rsi_score = 0.6  # 中性偏强
    elif 30 <= last['RSI'] < 40:
        rsi_score = 0.7  # 超卖区间，可能反弹
    elif last['RSI'] < 30:
        rsi_score = 0.8  # 严重超卖，反弹概率高
    elif 60 < last['RSI'] <= 70:
        rsi_score = 0.5  # 偏强但需警惕
    elif last['RSI'] > 70:
        rsi_score = 0.3  # 超买，风险较高
    scores.append(rsi_score)
    weights.append(0.15)

    # 4. KDJ 得分 (权重: 0.15)
    kdj_score = 0.5
    if last['K'] > last['D'] and last['J'] > last['K']:
        kdj_score = 0.75  # 金叉且J值最强
    elif last['K'] > last['D']:
        kdj_score = 0.65  # 金叉
    elif last['K'] < last['D'] and last['J'] < last['K']:
        kdj_score = 0.25  # 死叉且J值最弱
    elif last['K'] < last['D']:
        kdj_score = 0.35  # 死叉

    # KDJ 超买超卖调整
    if last['J'] < 0:
        kdj_score = min(kdj_score + 0.2, 0.9)  # J值超卖，加分
    elif last['J'] > 100:
        kdj_score = max(kdj_score - 0.2, 0.1)  # J值超买，减分
    scores.append(kdj_score)
    weights.append(0.15)

    # 5. 布林带位置得分 (权重: 0.15)
    bb_score = 0.5
    bb_width = last['BB_Upper'] - last['BB_Lower']
    bb_position = (last['Close'] - last['BB_Lower']) / bb_width if bb_width > 0 else 0.5

    if bb_position < 0.2:
        bb_score = 0.8  # 接近下轨，可能反弹
    elif bb_position > 0.8:
        bb_score = 0.3  # 接近上轨，可能回调
    elif 0.4 <= bb_position <= 0.6:
        bb_score = 0.6  # 中轨附近，相对安全
    scores.append(bb_score)
    weights.append(0.15)

    # 6. 成交量趋势得分 (权重: 0.10)
    vol_score = 0.5
    if len(df) >= 20:
        vol_ma5 = df['Volume'].iloc[-5:].mean()
        vol_ma20 = df['Volume'].iloc[-20:].mean()
        if vol_ma5 > vol_ma20 * 1.2:
            vol_score = 0.7  # 放量
        elif vol_ma5 < vol_ma20 * 0.8:
            vol_score = 0.4  # 缩量
    scores.append(vol_score)
    weights.append(0.10)

    # 计算加权平均
    total_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    return round(total_score, 4)


def predict_single_stock_no_transformer(code: str, name: str = "",
                                        strategy_config: Optional[Dict] = None  # 新增参数：策略配置
                                        ) -> Optional[Dict]:
    """预测单只股票（无Transformer版本）

    Args:
        code: 股票代码
        name: 股票名称

    Returns:
        预测结果字典，失败返回 None
    """
    if not _EF_AVAILABLE:
        logger.warning("efinance 不可用")
        return None

    try:
        # 获取历史数据
        df = ef.stock.get_quote_history(code, beg='20210101', end='20270101')
        if df is None or df.empty:
            return None
        if '日期' not in df.columns:
            return None

        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期').sort_index()

        # 转换列名
        temp = pd.DataFrame(index=df.index)
        temp['Open'] = df['开盘']
        temp['High'] = df['最高']
        temp['Low'] = df['最低']
        temp['Close'] = df['收盘']
        temp['Volume'] = df['成交量']
        temp['Turnover Rate'] = df['换手率']

        # 拦截检查
        skip, reason = should_intercept_stock(code, name, temp)
        if skip:
            logger.warning(f"[拦截-预测] 跳过 {code}: {reason}")
            return None

        # 计算传统因子
        temp = calculate_traditional_factors(temp)
        if len(temp) < 30:
            return None

        # 计算综合得分
        # 【修改处】：根据是否传入策略配置，选择不同的计算方式
        if strategy_config and 'weights' in strategy_config:
            # 使用回测优化后的权重进行加权得分计算
            # 注意：这里需要实现一个支持自定义权重的得分函数
            # 假设 calculate_factor_score 暂时不支持自定义权重，
            # 实际项目中应重构 calculate_factor_score 使其接受 weights 参数。
            # 这里演示一个简单的替换逻辑：
            score = calculate_custom_factor_score(temp, strategy_config['weights'])
        else:
            # 默认逻辑
            score = calculate_factor_score(temp)
        # 计算预测收益（基于ATR）
        last = temp.iloc[-1]
        atr = last['ATR'] if not pd.isna(last['ATR']) else last['Close'] * 0.02
        pred_ret = (atr / last['Close']) * (1 if score > 0.5 else -1)

        # 趋势判断
        trend = "上涨" if score > 0.55 else ("下跌" if score < 0.45 else "震荡")
        confidence = abs(score - 0.5) * 2  # 转换为 0-1 的置信度

        # 风险标志
        risk_flag = "正常"
        if last['RSI'] > 70 or last['J'] > 100:
            risk_flag = "⚠️ 超买风险"
        elif last['RSI'] < 30 or last['J'] < 0:
            risk_flag = "⚠️ 超卖机会"

        return {
            'code': code,
            'name': name,
            'trend': trend,
            'probability': round(float(confidence), 4),
            'predicted_ret': round(float(pred_ret * 100), 4),  # 百分比
            'factor_score': round(float(score), 4),
            'risk_flag': risk_flag,
            'method': 'traditional_factors',
            'last_close': round(float(last['Close']), 2),
            'rsi': round(float(last['RSI']), 2),
            'macd': round(float(last['MACD']), 4),
            'kdj_j': round(float(last['J']), 2),
        }

    except Exception as e:
        logger.warning(f"{code} 预测失败: {e}")
        return None


def predict_stocks_no_transformer(target_codes: List[str], code_to_name: Dict[str, str] = None) -> pd.DataFrame:
    """预测多只股票（无Transformer版本）

    Args:
        target_codes: 股票代码列表
        code_to_name: 代码到名称的映射

    Returns:
        预测结果 DataFrame，按 factor_score 降序排列
    """
    predictions = []
    code_to_name = code_to_name or {}
    # 【新增】：加载回测优化后的策略
    strategy_map = {}
    strategy_file = "./stock_cache/no_transformer_results/backtest_results.json"
    if os.path.exists(strategy_file):
        with open(strategy_file, 'r', encoding='utf-8') as f:
            strategies = json.load(f)
            # 构建代码到策略的映射
            strategy_map = {s['code']: s for s in strategies}
        logger.info(f"成功加载 {len(strategy_map)} 个优化策略")
    else:
        logger.warning("未找到回测策略文件，使用默认参数预测")
    for i, code in enumerate(tqdm(target_codes, desc="预测进度")):
        try:
            time.sleep(15)  # 避免请求过快

            name = code_to_name.get(code, "")
            config = strategy_map.get(code, {})
            result = predict_single_stock_no_transformer(code, name, strategy_config=config)

            if result:
                predictions.append(result)

        except Exception as e:
            logger.warning(f"{code} 预测失败: {e}")

    if predictions:
        df_result = pd.DataFrame(predictions)
        df_result = df_result.sort_values(by='factor_score', ascending=False).reset_index(drop=True)
        df_result.insert(0, 'rank', df_result.index + 1)
        return df_result

    return pd.DataFrame()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="集成预测（无Transformer版本）")
    parser.add_argument('--codes', nargs='+', default=None, help='指定股票代码列表')
    parser.add_argument('--pool', action='store_true', help='使用 stock_pool.json 中的股票池')
    parser.add_argument('--top', type=int, default=30, help='输出前N只股票')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--notify', action='store_true', help='发送企业微信通知')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    settings = get_settings()

    print("\n" + "=" * 70)
    print(f"集成预测（无Transformer版本） - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 确定目标代码
    target_codes = args.codes
    code_to_name = {}

    if args.pool:
        pool_file = settings.paths.stock_pool_file
        if os.path.exists(pool_file):
            with open(pool_file, 'r', encoding='utf-8') as f:
                pool_data = json.load(f)
            pool_codes = list(pool_data.get('default_pool', {}).keys())
            code_to_name = {v: k for k, v in pool_data.get('default_pool', {}).items()}
            if target_codes:
                target_codes = list(set(target_codes) | set(pool_codes))
            else:
                target_codes = pool_codes
            print(f"✓ 从股票池加载 {len(pool_codes)} 只股票")
        else:
            print(f"✗ 股票池文件不存在: {pool_file}")

    if not target_codes:
        # 使用默认列表
        target_codes = list(STOCK_CODES.values())
        code_to_name = {v: k for k, v in STOCK_CODES.items()}

        # 股票池拦截
        name_to_code_map = {n: c for n, c in STOCK_CODES.items() if c in target_codes}
        clean_map = filter_codes_by_name(name_to_code_map)
        target_codes = list(clean_map.values())
        code_to_name = {v: k for k, v in clean_map.items()}
        print(f"✓ 使用默认股票池: {len(target_codes)} 只")

    print(f"  目标股票数: {len(target_codes)}")
    print(f"  输出前 {args.top} 只")
    print("=" * 70)

    # 执行预测
    results = predict_stocks_no_transformer(target_codes, code_to_name)

    if results.empty:
        print("没有成功预测任何股票")
        return

    # 输出结果
    top_results = results.head(args.top)
    print(
        f"\n{'排名':>4} {'代码':<8} {'名称':<10} {'趋势':<6} {'置信度':>8} {'预测收益':>10} {'因子得分':>8} {'风险标志':<12}")
    print("-" * 80)
    for _, row in top_results.iterrows():
        print(f"{int(row['rank']):>4} {row['code']:<8} {row.get('name', ''):<10} {row['trend']:<6} "
              f"{row['probability']:>8.1%} {row['predicted_ret']:>+9.2f}% "
              f"{row['factor_score']:>8.4f} {row['risk_flag']:<12}")

    # 保存到独立目录
    output_dir = "./stock_cache/no_transformer_results"
    os.makedirs(output_dir, exist_ok=True)

    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(output_dir, "stock_predictions_no_transformer.json")

    results.to_json(output_file, orient='records', force_ascii=False, indent=2)
    print(f"\n✓ 预测结果已保存: {output_file} ({len(results)} 只股票)")

    # 通知
    if args.notify and settings.paths.wechat_webhook:
        try:
            import requests
            summary = f"📊 集成预测完成（无Transformer版本）\n共 {len(results)} 只股票\n\n"
            summary += "TOP 10:\n"
            for _, row in results.head(10).iterrows():
                summary += f"  {row['code']} {row.get('name', '')} {row['trend']} 得分{row['factor_score']:.2f}\n"

            headers = {'Content-Type': 'application/json'}
            data = {"msgtype": "text", "text": {"content": summary}}
            requests.post(settings.paths.wechat_webhook, headers=headers, data=json.dumps(data), timeout=5)
        except Exception as e:
            print(f"通知发送失败: {e}")


if __name__ == "__main__":
    main()

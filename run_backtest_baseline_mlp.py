# -*- coding: utf-8 -*-
"""
run_backtest_baseline_mlp.py

日线 MLP 基线回测完整脚本 (修正版 V2)。
修复：
1. visualize_backtest_with_split 补全 market_data 参数
2. RiskManager 传参修正 settings→settings.risk
3. STOCK_CODES 遍历改用 .values() 并正确匹配数据
4. 去除重复的 prepare_stock_data 调用
5. calculate_comprehensive_stats 传参修正
6. params 补全所有必要字段
"""

import os
import logging
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import get_settings, STOCK_CODES
from data import check_and_clean_cache, save_pickle_cache, load_pickle_cache
from data.indicators_new import (
    prepare_stock_data,
    calculate_orthogonal_factors_without_transformer
)
from data.loader_new import download_market_data, download_stocks_data
from data.types import TRADITIONAL_FACTOR_COLS
from backtest.engine_no_transformer_new import run_backtest_loop_no_transformer
from backtest.evaluator import calculate_comprehensive_stats
from backtest.visualizer import visualize_backtest_with_split
from risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("backtest_baseline_mlp.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

output_dir = "./stock_cache/baseline_mlp_results"
os.makedirs(output_dir, exist_ok=True)
viz_dir = os.path.join(output_dir, "charts")
os.makedirs(viz_dir, exist_ok=True)
model_dir = os.path.join(output_dir, "models")
os.makedirs(model_dir, exist_ok=True)


# ==================== MLP 模型定义 ====================

class BaselineMLP(nn.Module):
    """简单 MLP 回归模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_mlp_model(
        X_train, y_train, X_val, y_val, input_dim,
        hidden_dim=64, num_layers=2, dropout=0.1,
        lr=1e-3, batch_size=256, epochs=30,
        device="cuda" if torch.cuda.is_available() else "cpu",
        early_stop_patience=5,
) -> nn.Module:
    model = BaselineMLP(input_dim, hidden_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            logger.info(f"[MLP 训练] epoch={epoch + 1}/{epochs}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def prepare_mlp_features_from_df(
        df: pd.DataFrame,
        lookback: int = 120,
        horizon: int = 5,
        standardize: bool = True,
) -> tuple:
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return_1d', 'score', 'future_return']
    feature_cols = [col for col in df.columns if
                    col not in exclude_cols and df[col].dtype in [np.float32, np.float64, np.int32, np.int64]]

    X = df[feature_cols].values.copy()

    future_close = df['Close'].shift(-horizon)
    ret = (future_close - df['Close']) / df['Close']
    y = ret.values

    if standardize:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        X = (X - mean) / (std + 1e-8)

    valid_mask = ~np.isnan(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    dates = df.index[valid_mask]

    if lookback > 0 and len(X) > lookback:
        X = X[lookback:]
        y = y[lookback:]
        dates = dates[lookback:]

    return X, y, dates, feature_cols


def run_single_stock_backtest_baseline_mlp(
        stock_code: str,
        stock_name: str,
        df: pd.DataFrame,
        market_data: pd.DataFrame,
        stocks_data: dict,
        params: dict,
        mlp_config: dict
) -> tuple:
    """对单只股票执行 MLP 基线回测"""
    X, y, dates, feature_cols = prepare_mlp_features_from_df(
        df,
        lookback=mlp_config['lookback'],
        horizon=mlp_config['horizon'],
        standardize=mlp_config['standardize'],
    )

    if len(X) < 200:
        logger.warning(f"[{stock_code}] 样本数 {len(X)} < 200，跳过 MLP 回测")
        return None, None, df

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    input_dim = X.shape[1]
    model = train_mlp_model(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        hidden_dim=mlp_config['hidden_dim'],
        num_layers=mlp_config['num_layers'],
        dropout=mlp_config['dropout'],
        lr=mlp_config['lr'],
        batch_size=mlp_config['batch_size'],
        epochs=mlp_config['epochs'],
        device=mlp_config['device'],
        early_stop_patience=mlp_config['early_stop_patience'],
    )

    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(mlp_config['device'])
        score_array = model(X_tensor).cpu().numpy()

    df["score"] = np.nan
    score_dates = dates[-len(score_array):]
    df.loc[score_dates, "score"] = score_array

    weights = {}
    trades_df, stats, df = run_backtest_loop_no_transformer(
        df, stock_code, market_data, weights, params,
        regime=mlp_config.get('regime', None),
        stocks_data=stocks_data,
        initial_capital=mlp_config.get('initial_capital', 100000.0),
    )

    if trades_df is not None:
        model_path = os.path.join(model_dir, f"{stock_code}_mlp.pt")
        torch.save(model.state_dict(), model_path)

        try:
            chart_path = os.path.join(viz_dir, f"{stock_code}_backtest.png")
            split_date = dates[split_idx]
            if split_date in df.index:
                split_idx_in_df = df.index.get_loc(split_date)
            else:
                split_idx_in_df = int(len(df) * 0.8)

            # ★ 修复：补全 market_data 参数
            visualize_backtest_with_split(
                df, trades_df,
                stock_name=stock_name,
                split_idx=split_idx_in_df,
                market_data=market_data,
                save_path=chart_path,
            )
        except Exception as e:
            logger.warning(f"[{stock_code}] 可视化生成失败: {e}")

    return trades_df, stats, df


# ==================== 多进程 ====================

_worker_market_data = None
_worker_stocks_data = None
_worker_params = None
_worker_mlp_config = None


def _init_worker(market_data, stocks_data, params, mlp_config):
    global _worker_market_data, _worker_stocks_data, _worker_params, _worker_mlp_config
    _worker_market_data = market_data
    _worker_stocks_data = stocks_data
    _worker_params = params
    _worker_mlp_config = mlp_config


def _process_single_stock(stock_name: str):
    """多进程回调"""
    global _worker_market_data, _worker_stocks_data, _worker_params, _worker_mlp_config

    # ★ 修复：用股票名称从数据字典中获取数据
    df = _worker_stocks_data.get(stock_name)
    if df is None or len(df) < 60:
        return None

    # 获取股票代码
    stock_code = STOCK_CODES.get(stock_name, stock_name)

    trades_df, stats, df = run_single_stock_backtest_baseline_mlp(
        stock_code, stock_name, df,
        _worker_market_data,
        _worker_stocks_data,
        params=_worker_params,
        mlp_config=_worker_mlp_config,
    )

    if trades_df is None:
        return None

    return {
        "code": stock_code,
        "name": stock_name,
        "trades_df": trades_df,
        "stats": stats,
        "df": df,
    }


def run_baseline_mlp_backtest():
    """主函数"""
    settings = get_settings()
    # ★ 修复：RiskManager(settings) → RiskManager(settings.risk)
    risk_manager = RiskManager(settings.risk)

    # 1. 大盘数据
    market_cache_file = "./stock_cache/no_transformer_market_data.pkl"
    try:
        if not check_and_clean_cache(market_cache_file):
            market_data = download_market_data()
            save_pickle_cache(market_cache_file, market_data)
        else:
            market_data = load_pickle_cache(market_cache_file)
    except Exception as e:
        print(f"警告: 大盘数据加载失败: {e}")
        if os.path.exists(market_cache_file):
            market_data = load_pickle_cache(market_cache_file)
        else:
            print("错误: 没有可用的大盘数据")
            return
    if market_data is None or len(market_data) == 0:
        logger.error("大盘数据下载失败")
        return

    # 2. 个股数据
    print("\n[2/3] 检查个股数据...")
    stocks_cache_file = "./stock_cache/no_transformer_stocks_data.pkl"
    if not check_and_clean_cache(stocks_cache_file):
        print("下载股票数据...")
        stocks_data = download_stocks_data(STOCK_CODES)
        save_pickle_cache(stocks_cache_file, stocks_data)
    else:
        print("使用缓存的股票数据...")
        stocks_data = load_pickle_cache(stocks_cache_file)
    if not stocks_data:
        print("错误: 无法获取个股数据")
        return

    # 数据验证和修复
    first_key = list(stocks_data.keys())[0] if stocks_data else None
    if first_key and first_key in ['stocks_data', 'last_date']:
        print("检测到错误的数据结构，正在修复...")
        if isinstance(stocks_data, dict) and 'stocks_data' in stocks_data:
            stocks_data = stocks_data['stocks_data']

    if not stocks_data or len(stocks_data) == 0:
        print("错误: 无法获取有效的股票数据")
        return

    # ★ 修复：只调用一次 prepare_stock_data
    logger.info("计算收益率和技术指标...")
    stocks_data = prepare_stock_data(stocks_data)
    logger.info("计算正交因子...")
    for name, df in stocks_data.items():
        stocks_data[name] = calculate_orthogonal_factors_without_transformer(df)

    # ★ 修复：params 补全所有必要字段
    params = {
        "bull": {
            "buy_threshold": 0.6,
            "sell_threshold": -0.2,
            "stop_loss": -0.08,
            "hold_days": 15,
            "trailing_profit_level1": 0.06,
            "trailing_profit_level2": 0.12,
            "trailing_drawdown_level1": 0.08,
            "trailing_drawdown_level2": 0.04,
            "take_profit_multiplier": 3.0,
        },
        "bear": {
            "buy_threshold": 0.7,
            "sell_threshold": -0.3,
            "stop_loss": -0.10,
            "hold_days": 10,
            "trailing_profit_level1": 0.05,
            "trailing_profit_level2": 0.10,
            "trailing_drawdown_level1": 0.06,
            "trailing_drawdown_level2": 0.03,
            "take_profit_multiplier": 2.5,
        },
        "neutral": {
            "buy_threshold": 0.65,
            "sell_threshold": -0.25,
            "stop_loss": -0.09,
            "hold_days": 12,
            "trailing_profit_level1": 0.05,
            "trailing_profit_level2": 0.10,
            "trailing_drawdown_level1": 0.07,
            "trailing_drawdown_level2": 0.03,
            "take_profit_multiplier": 2.8,
        },
        # ★ 新增：5种市场状态兼容
        "strong_bull": {
            "buy_threshold": 0.55,
            "sell_threshold": -0.15,
            "stop_loss": -0.06,
            "hold_days": 20,
            "trailing_profit_level1": 0.08,
            "trailing_profit_level2": 0.15,
            "trailing_drawdown_level1": 0.10,
            "trailing_drawdown_level2": 0.05,
            "take_profit_multiplier": 3.5,
        },
        "weak": {
            "buy_threshold": 0.70,
            "sell_threshold": -0.30,
            "stop_loss": -0.07,
            "hold_days": 10,
            "trailing_profit_level1": 0.04,
            "trailing_profit_level2": 0.08,
            "trailing_drawdown_level1": 0.05,
            "trailing_drawdown_level2": 0.02,
            "take_profit_multiplier": 2.0,
        },
    }

    mlp_config = {
        "lookback": 120,
        "horizon": 5,
        "standardize": True,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 30,
        "early_stop_patience": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "regime": None,
        "initial_capital": 100000.0,
    }

    # ★ 修复：用股票名称列表遍历（与 stocks_data 的 key 一致）
    stock_names = [name for name in STOCK_CODES.keys() if name in stocks_data]
    logger.info(f"有效股票数: {len(stock_names)}")

    num_workers = max(1, cpu_count() - 1)
    with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(market_data, stocks_data, params, mlp_config),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(_process_single_stock, stock_names),
                total=len(stock_names),
                desc="Baseline MLP 回测",
            )
        )

    # 汇总结果
    all_trades = []
    all_stats = []
    for res in results:
        if res is None:
            continue
        stats = res.get("stats")
        trades = res.get("trades_df")
        if stats:
            entry = {"code": res["code"], "name": res.get("name", "")}
            if isinstance(stats, dict):
                entry.update(stats)
            all_stats.append(entry)
        if trades is not None and len(trades) > 0:
            all_trades.append(trades)

    if not all_stats:
        logger.warning("没有有效回测结果")
        return

    stats_df = pd.DataFrame(all_stats)

    # ★ 修复：对合并后的 trades_df 计算综合统计（而非对 stats_df）
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        comprehensive_stats = calculate_comprehensive_stats(combined_trades)
        logger.info("=== MLP 基线综合统计 ===")
        for k, v in comprehensive_stats.items():
            if v is not None:
                logger.info(f"  {k}: {v}")

    # 打印每只股票的结果
    print(f"\n{'名称':<10} {'收益%':>8} {'胜率%':>7} {'交易':>5} {'夏普':>7}")
    print("-" * 45)
    for _, row in stats_df.iterrows():
        print(
            f"{row.get('name', ''):<10} "
            f"{row.get('total_return', 0):>8.2f} "
            f"{row.get('win_rate', 0):>7.1f} "
            f"{row.get('total_trades', 0):>5} "
            f"{row.get('sharpe_ratio', 0):>7.2f}"
        )

    # 保存结果
    stats_path = os.path.join(output_dir, "baseline_mlp_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"统计结果已保存: {stats_path}")


if __name__ == "__main__":
    run_baseline_mlp_backtest()

# -*- coding: utf-8 -*-
"""
数据下载入口脚本
下载全量A股历史数据并生成训练用 feather 文件
从 TransformerStock.py 的数据获取逻辑提取
"""
import os
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import talib as ta
from tqdm import tqdm

from config import get_settings, STOCK_CODES
from data.indicators_no_transformer import safe_sma

logger = logging.getLogger(__name__)

try:
    import efinance as ef
except ImportError:
    ef = None
    logger.error("efinance 未安装")


FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover Rate',
    'MA5', 'MA10', 'MA20', 'MACD', 'K', 'D', 'J', 'RSI', 'ADX',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'OBV', 'CCI',
]


def get_all_a_stock_codes() -> list:
    """获取有效的A股代码列表"""
    try:
        df = ef.stock.get_realtime_quotes()
        if df is not None and not df.empty:
            codes = df['股票代码'].tolist()
            # 过滤：排除ST、退市等
            valid_codes = [c for c in codes if not any(x in str(c) for x in ['ST', '退'])]
            logger.info(f"获取到 {len(valid_codes)} 只A股代码")
            return valid_codes
    except Exception as e:
        logger.error(f"获取A股代码列表失败: {e}")
    return []


def process_single_batch(batch_codes: list, start_date: str, end_date: str) -> pd.DataFrame:
    """处理一批股票的数据下载和技术指标计算"""
    all_data = []
    failed = []

    for code in tqdm(batch_codes, desc="下载批次", leave=False):
        try:
            time.sleep(10)
            df = ef.stock.get_quote_history(code, beg=start_date, end=end_date)

            if df is None or df.empty:
                failed.append(code)
                continue

            temp = pd.DataFrame()
            temp['Date'] = pd.to_datetime(df['日期'])
            temp['Open'] = df['开盘']
            temp['High'] = df['最高']
            temp['Low'] = df['最低']
            temp['Close'] = df['收盘']
            temp['Volume'] = df['成交量']
            temp['Turnover Rate'] = df['换手率']
            temp['Code'] = code

            # 技术指标
            temp['MA5'] = safe_sma(temp['Close'], period=5)
            temp['MA10'] = safe_sma(temp['Close'], period=10)
            temp['MA20'] = safe_sma(temp['Close'], period=20)
            temp['MACD'], temp['MACD_Signal'], temp['MACD_Hist'] = ta.MACD(
                temp['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            temp['K'], temp['D'] = ta.STOCH(
                temp['High'], temp['Low'], temp['Close'],
                fastk_period=9, slowk_period=3, slowd_period=3)
            temp['J'] = 3 * temp['K'] - 2 * temp['D']
            temp['RSI'] = ta.RSI(temp['Close'], timeperiod=14)
            temp['ADX'] = ta.ADX(temp['High'], temp['Low'], temp['Close'], timeperiod=14)
            temp['BB_Upper'], temp['BB_Middle'], temp['BB_Lower'] = ta.BBANDS(
                temp['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            temp['OBV'] = ta.OBV(temp['Close'], temp['Volume'])
            temp['CCI'] = ta.CCI(temp['High'], temp['Low'], temp['Close'], timeperiod=20)

            all_data.append(temp)

        except Exception as e:
            failed.append(code)
            logger.debug(f"{code} 下载失败: {e}")

    if failed:
        logger.warning(f"批次失败: {len(failed)} 只")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="数据下载")
    parser.add_argument('--codes', nargs='+', default=None, help='指定股票代码（默认使用配置中的股票池）')
    parser.add_argument('--all', action='store_true', help='下载全量A股数据')
    parser.add_argument('--start', type=str, default='20210101', help='开始日期')
    parser.add_argument('--end', type=str, default='20270101', help='结束日期')
    parser.add_argument('--batch-size', type=int, default=30, help='每批处理数量')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    settings = get_settings()

    output_path = args.output or settings.paths.stock_data_file

    if ef is None:
        logger.error("efinance 未安装，无法下载数据")
        return

    print("\n" + "=" * 60)
    print(f"数据下载 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 确定代码列表
    if args.codes:
        codes = args.codes
    elif args.all:
        codes = get_all_a_stock_codes()
        if not codes:
            logger.error("无法获取A股代码列表")
            return
    else:
        codes = list(STOCK_CODES.values())

    print(f"  目标股票数: {len(codes)}")
    print(f"  日期范围: {args.start} ~ {args.end}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  输出文件: {output_path}")
    print("=" * 60)

    # 检查是否已有部分数据
    batch_dir = os.path.join(os.path.dirname(output_path), "stock_batches")
    os.makedirs(batch_dir, exist_ok=True)

    batches = [codes[i:i + args.batch_size] for i in range(0, len(codes), args.batch_size)]
    total_batches = len(batches)
    batch_files = []

    for i, batch in enumerate(tqdm(batches, desc="处理批次")):
        batch_file = os.path.join(batch_dir, f"batch_{i + 1}_{total_batches}.feather")
        batch_files.append(batch_file)

        if os.path.exists(batch_file):
            logger.info(f"批次 {i + 1}/{total_batches} 已存在，跳过")
            continue

        try:
            logger.info(f"处理批次 {i + 1}/{total_batches} ({len(batch)} 只)")
            batch_data = process_single_batch(batch, args.start, args.end)

            if batch_data is not None and not batch_data.empty:
                batch_data.to_feather(batch_file)
                logger.info(f"批次 {i + 1} 已保存 ({len(batch_data)} 条)")
            else:
                logger.warning(f"批次 {i + 1} 无有效数据")
        except Exception as e:
            logger.error(f"批次 {i + 1} 处理失败: {e}")

    # 合并
    print("\n合并批次文件...")
    all_data = []
    for bf in batch_files:
        if os.path.exists(bf):
            all_data.append(pd.read_feather(bf))

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"合并完成: {len(combined)} 条记录")

        # 基础清洗
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in combined.columns:
                combined = combined[combined[col] > 0.01]
        if 'Volume' in combined.columns:
            combined = combined[combined['Volume'] > 0]

        # 删除 NaN
        combined = combined.dropna(subset=FEATURES)
        combined = combined.reset_index(drop=True)

        print(f"清洗后: {len(combined)} 条记录")
        print(f"股票数: {combined['Code'].nunique()}")
        print(f"日期范围: {combined['Date'].min()} ~ {combined['Date'].max()}")

        combined.to_feather(output_path)
        print(f"\n✓ 数据已保存: {output_path}")
    else:
        print("✗ 没有有效数据")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
数据加载模块（增强版）
封装 efinance / akshare 的数据获取逻辑，统一错误处理和重试机制
新增：数据完整性检查、缓存校验、异常捕获与恢复
"""
import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np
from pandas import DataFrame

from config import get_settings
from data.cache import _clean_stock_data
from exceptions import DataFetchError, DataValidationError

logger = logging.getLogger(__name__)

# 延迟导入，避免未安装时报错
try:
    import efinance as ef

    _EF_AVAILABLE = True
except ImportError:
    _EF_AVAILABLE = False
    ef = None
    logger.warning("efinance 未安装，个股数据下载功能不可用")

try:
    import akshare as ak

    _AK_AVAILABLE = True
except ImportError:
    _AK_AVAILABLE = False
    ak = None
    logger.warning("akshare 未安装，大盘数据下载功能不可用")

# ==================== 新增：数据完整性检查常量 ====================
MIN_DATA_ROWS = 60  # 最小数据行数
MAX_NAN_RATIO = 0.1  # 最大缺失值比例


def _setup_proxy():
    """设置代理环境变量"""
    proxy = os.getenv('HTTP_PROXY', '') or os.getenv('HTTPS_PROXY', '')
    if proxy:
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy


# ==================== 新增：数据完整性检查函数 ====================
import pandas as pd
import numpy as np
from typing import Tuple
import logging

# 假设这些常量在文件其他地方已定义
MIN_DATA_ROWS = 20
MAX_NAN_RATIO = 0.1

logger = logging.getLogger(__name__)


def validate_data_integrity(
        df: pd.DataFrame,
        code: str = "",
        name: str = "",
        check_nan_ratio: bool = True,
        check_price_validity: bool = True,
) -> tuple[None, bool, str] | tuple[DataFrame, bool, str]:
    """数据完整性检查与清洗

    不仅检查数据问题，还会尝试修复/过滤非正常值（如价格<=0）。
    对于价格非正值，采用 '置为NaN -> 前向填充' 的策略修复。

    Args:
        df: 待检查的DataFrame
        code: 股票代码（用于日志）
        name: 股票名称（用于日志）
        check_nan_ratio: 是否检查缺失值比例
        check_price_validity: 是否检查并清洗价格有效性

    Returns:
        (清洗后的DataFrame, 是否通过检查, 错误信息)
        如果检查未通过，返回的DataFrame可能为None或原始DataFrame。
    """
    if df is None:
        return None, False, "数据为None"

    # 为了避免修改原数据，创建一个副本进行操作
    df_clean = df.copy()

    if len(df_clean) == 0:
        return None, False, "数据为空"

    # 检查数据行数
    if len(df_clean) < MIN_DATA_ROWS:
        return None, False, f"数据行数不足: {len(df_clean)} < {MIN_DATA_ROWS}"

    # ---------------------------------------------------------
    # 1. 检查并清洗价格有效性 (核心修改部分)
    # ---------------------------------------------------------
    if check_price_validity:
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df_clean.columns:
                invalid_mask = df_clean[col] <= 0
                invalid_count = invalid_mask.sum()

                if invalid_count > 0:
                    logger.warning(f"[{code}] {col} 存在 {invalid_count} 个非正值")
                    # 【关键修改】：不要 ffill，直接找到第一个有效值的索引，截断前面的无效数据
                    first_valid_idx = df_clean[col][df_clean[col] > 0].first_valid_index()
                    if first_valid_idx is not None:
                        df_clean = df_clean.loc[first_valid_idx:]
                    else:
                        return None, False, f"{col} 全部为非正值，无有效数据"
    # ---------------------------------------------------------
    # 2. 检查缺失值比例 (在清洗后再次检查)
    # ---------------------------------------------------------
    if check_nan_ratio:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in df_clean.columns:
                nan_ratio = df_clean[col].isna().sum() / len(df_clean)
                # 如果清洗后缺失值依然过高，说明数据质量太差
                if nan_ratio > MAX_NAN_RATIO:
                    return None, False, f"{col} 缺失比例过高(清洗后): {nan_ratio:.1%}"

    # ---------------------------------------------------------
    # 3. 最终兜底检查
    # ---------------------------------------------------------
    # 如果经过清洗，关键列依然存在NaN（且无法填充），则视为不通过
    final_nan_check = df_clean[['Open', 'High', 'Low', 'Close']].isna().sum().sum()
    if final_nan_check > 0:
        # 可以选择dropna，或者直接报错。这里选择报错更安全，防止隐藏数据断档问题
        # 如果希望强制通过，可以在这里执行 df_clean.dropna(subset=required_cols, inplace=True)
        return None, False, "清洗后仍存在无效数据，请检查源数据连续性"

    return df_clean, True, "OK"


def clean_and_validate_data(
        df: pd.DataFrame,
        code: str = "",
        force_datetime_index: bool = True,
        remove_duplicates: bool = True,
        fill_nan_method: str = 'ffill',
) -> pd.DataFrame:
    """数据清洗和验证工具函数

    Args:
        df: 待清洗的DataFrame
        code: 股票代码（用于日志）
        force_datetime_index: 是否强制转换为DatetimeIndex
        remove_duplicates: 是否移除重复索引
        fill_nan_method: 缺失值填充方法 ('ffill', 'bfill', 'mean', 'median')

    Returns:
        清洗后的DataFrame
    """
    df = df.copy()

    # 强制转换为DatetimeIndex
    if force_datetime_index:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.warning(f"[{code}] 索引转换为DatetimeIndex失败: {e}")

    # 排序
    df = df.sort_index()

    # 移除重复索引
    if remove_duplicates and df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        logger.debug(f"[{code}] 移除 {dup_count} 个重复索引")
        df = df[~df.index.duplicated(keep='last')]

    # 缺失值处理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            nan_count = df[col].isna().sum()
            if fill_nan_method == 'ffill':
                df[col] = df[col].ffill().bfill()
            elif fill_nan_method == 'bfill':
                df[col] = df[col].bfill().ffill()
            elif fill_nan_method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif fill_nan_method == 'median':
                df[col] = df[col].fillna(df[col].median())
            logger.debug(f"[{code}] {col} 使用 {fill_nan_method} 填充 {nan_count} 个缺失值")

    return df


def download_market_data(
        retries: int = 3,
        delay: float = 1.0,
) -> pd.DataFrame:
    """下载大盘数据（上证指数）- 增强版

    新增：数据完整性检查、重试机制优化
    """
    _setup_proxy()

    if not _AK_AVAILABLE:
        raise DataFetchError("akshare 未安装，无法下载大盘数据")

    for attempt in range(retries):
        try:
            logger.info(f"下载大盘数据:(尝试 {attempt + 1}/{retries})")

            df = ak.stock_zh_index_daily(symbol="sh000001")
            df = df.rename(columns={
                'date': 'Date', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()

            # 新增：数据完整性检查
            df, is_valid, msg = validate_data_integrity(df, code="market")
            if not is_valid:
                logger.warning(f"大盘数据完整性检查失败: {msg}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue

            logger.info(f"大盘数据下载成功: {len(df)} 行")
            return df

        except Exception as e:
            logger.error(f"下载大盘数据失败 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    raise DataFetchError(f"下载大盘数据失败，已重试 {retries} 次")


def download_stock_data(
        code: str,
        retries: int = 3,
        delay: float = 1.0,
) -> pd.DataFrame:
    """下载单只股票数据 - 增强版

    新增：数据完整性检查、异常捕获
    """
    _setup_proxy()

    if not _EF_AVAILABLE:
        raise DataFetchError("efinance 未安装，无法下载个股数据")

    for attempt in range(retries):
        try:
            logger.debug(f"隔15秒下载股票 {code} 数据:")
            time.sleep(15)
            df = ef.stock.get_quote_history(
                code,
                klt=101, fqt=1
            )

            if df is None or len(df) == 0:
                raise DataFetchError(f"股票 {code} 数据为空")

            df = df.rename(columns={
                '日期': 'Date', '开盘': 'Open', '最高': 'High',
                '最低': 'Low', '收盘': 'Close', '成交量': 'Volume',
                '换手率': 'Turnover Rate',
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()

            # 新增：数据清洗
            df = clean_and_validate_data(df, code=code)

            # 新增：数据完整性检查
            df, is_valid, msg = validate_data_integrity(df, code=code)
            if not is_valid:
                logger.warning(f"股票 {code} 数据完整性检查失败: {msg}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue

            return df

        except Exception as e:
            logger.error(f"下载股票 {code} 数据失败 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    raise DataFetchError(f"下载股票 {code} 数据失败，已重试 {retries} 次")


def download_stocks_data(
        codes: list,
        max_workers: int = 1,
) -> Dict[str, pd.DataFrame]:
    """批量下载多只股票数据 - 增强版

    新增：异常捕获、进度日志
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    failed = []

    logger.info(f"开始下载 {len(codes)} 只股票数据...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_stock_data, code): code
            for code in codes
        }

        for future in as_completed(futures):
            code = futures[future]
            try:
                df = future.result()
                # ===== 加上这一行 =====
                df = _clean_stock_data(df)
                # =======================
                if df is not None and len(df) > 0:
                    results[code] = df
                else:
                    failed.append(code)
                    logger.warning(f"股票 {code} 数据为空")
            except Exception as e:
                failed.append(code)
                logger.error(f"下载股票 {code} 失败: {e}")

    logger.info(f"下载完成: 成功 {len(results)} 只, 失败 {len(failed)} 只")

    if failed:
        logger.warning(f"失败的股票: {failed}")

    return results

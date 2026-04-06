# -*- coding: utf-8 -*-
"""
缓存管理模块
Pickle缓存 + Transformer因子缓存，支持原子写入和竞态保护
"""
import os
import time
import pickle
import logging
from datetime import datetime, date as date_type, timedelta
from typing import Optional, Any, Tuple

import pandas as pd

from config import get_settings
from exceptions import CacheIOError

logger = logging.getLogger(__name__)


def get_trading_day_status() -> Tuple[bool, str]:
    """判断当前是否可能有最新的交易数据"""
    now = datetime.now()
    weekday = now.weekday()
    if weekday in [5, 6]:
        return False, "周末休市"
    if 9 <= now.hour <= 15:
        return True, "交易日盘中"
    return False, "交易日盘前/收盘后"


def check_and_clean_cache(cache_file: str) -> bool:
    """检查缓存文件是否有效，返回 True 表示缓存可用"""
    try:
        if not os.path.exists(cache_file):
            return False

        with open(cache_file, 'rb') as f:
            data = pickle.load(f)

        # ============ 关键修改：兼容 dict[str, DataFrame] 结构 ============
        if isinstance(data, dict):
            # 情况1：错误结构 {"stocks_data": {...}, "last_date": ...}
            first_key = list(data.keys())[0]
            if first_key in ['stocks_data', 'last_date', 'Date']:
                os.remove(cache_file)
                logger.warning(f"检测到错误的缓存结构，已删除: {cache_file}")
                return False

            # 情况2：正确结构 dict[str, DataFrame]，取第一个 DataFrame 的最后日期
            first_value = data[first_key]
            if isinstance(first_value, pd.DataFrame):
                last_date = first_value.index[-1]
            else:
                os.remove(cache_file)
                logger.warning(f"缓存值类型异常: {type(first_value)}，已删除: {cache_file}")
                return False
        elif isinstance(data, pd.DataFrame):
            # 情况3：单个 DataFrame（大盘数据）
            last_date = data.index[-1]
        else:
            os.remove(cache_file)
            logger.warning(f"未知缓存类型: {type(data)}，已删除: {cache_file}")
            return False
        # ==================================================================
        #2026/04/06今天暂时屏蔽
        return True

        # last_date = pd.to_datetime(last_date).date()
        # today = datetime.now().date()
        #
        # if last_date >= today - timedelta(days=1):
        #     logger.info(f"缓存有效: {cache_file} (最后日期: {last_date})")
        #     return True
        # else:
        #     logger.info(f"缓存过期: {cache_file} (最后日期: {last_date})")
        #     return False

    except Exception as e:
        logger.warning(f"缓存读取异常: {e}")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        return False



def load_pickle_cache(cache_path):
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        # 数据完整性检查
        if data is None or len(data) == 0:
            raise ValueError("缓存数据为空或加载失败")
        return data
    except Exception as e:
        logging.warning(f"加载缓存失败: {e}, 将重新下载数据...")
        return None

def validate_data_integrity(df, code="", name=""):
    """数据完整性检查"""
    if df is None or len(df) == 0:
        return False, "数据为空"
    if len(df) < 10:
        return False, f"数据行数不足: {len(df)} < {10}"
    # 检查缺失值比例
    nan_ratio = df.isna().sum().sum() / (len(df) * len(df.columns))
    if nan_ratio > 0.2:
        return False, f"缺失值比例过高: {nan_ratio:.1%}"
    return True, "OK"



def save_pickle_cache(cache_file: str, data: dict):
    """保存pickle缓存"""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        raise CacheIOError(f"写入失败: {e}", cache_path=cache_file)


# ==================== Transformer 因子缓存 ====================

def get_transformer_cache_path(code: str) -> str:
    """获取某只股票的 Transformer 缓存文件路径"""
    settings = get_settings()
    return os.path.join(settings.paths.cache_dir, f'transformer_factors_{code}.pkl')


def load_transformer_cache(code: str, current_df_last_date) -> Optional[pd.DataFrame]:
    """加载 Transformer 因子缓存

    如果缓存数据覆盖了当前需要的日期，则直接返回
    否则删除旧缓存返回 None
    """
    cache_path = get_transformer_cache_path(code)
    if not os.path.exists(cache_path):
        return None

    # 竞态保护：检查写入锁
    lock_path = cache_path + '.writing'
    if os.path.exists(lock_path):
        for _ in range(50):
            time.sleep(0.2)
            if not os.path.exists(lock_path):
                break
        else:
            return None

    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

        cache_last_date = _to_date(cache.get('last_date'))
        curr_date = _to_date(current_df_last_date)

        if cache_last_date is not None and curr_date is not None and cache_last_date >= curr_date:
            return cache.get('df')
        else:
            os.remove(cache_path)
            return None

    except Exception:
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return None


def save_transformer_cache(code: str, last_date, result_df: pd.DataFrame):
    """保存 Transformer 因子缓存（原子写入）"""
    cache_path = get_transformer_cache_path(code)
    lock_path = cache_path + '.writing'
    tmp_path = cache_path + '.tmp'

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with open(lock_path, 'w') as _:
            pass

        with open(tmp_path, 'wb') as f:
            pickle.dump({'last_date': last_date, 'df': result_df}, f)

        os.replace(tmp_path, cache_path)

    except Exception as e:
        logger.error(f"保存 {code} Transformer缓存失败: {e}")
    finally:
        for p in [lock_path, tmp_path]:
            if os.path.exists(p):
                os.remove(p)


def _to_date(value) -> Optional[date_type]:
    """统一转换为 date 类型"""
    if value is None:
        return None
    if isinstance(value, date_type) and not isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return pd.to_datetime(value).date()
    if hasattr(value, 'date'):
        return value.date()
    return None


def _clean_stock_data(df):
    """清洗个股数据：移除停牌日（价格为0的行）"""
    original_len = len(df)

    # 只保留 OHLC 都大于 0 的行（排除停牌日）
    mask = (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)
    df = df[mask].copy()

    cleaned_len = len(df)
    removed = original_len - cleaned_len

    if removed > 0:
        print(f"  清洗数据: 移除 {removed} 行非正值数据 ({original_len} → {cleaned_len})")

    return df

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """
    策略抽象基类

    所有具体策略必须继承此类并实现以下方法：
    - generate_signal(): 生成交易信号（单步，回测/实盘兼容）
    - get_param_space(): 定义 Optuna 参数空间
    - generate_signals_vectorized(): 生成向量化信号（建议实现，用于向量化回测）
    """

    name: str = "base"
    description: str = ""
    keep: bool = False  # 是否为保留策略（用于策略筛选）

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        params: Dict[str, Any],
        regime: str = "neutral",
    ) -> Dict[str, Any]:
        """生成交易信号

        Args:
            df: 包含因子和价格数据的 DataFrame
            idx: 当前时间步索引
            params: 策略参数（由 Optuna 优化得到）
            regime: 市场状态

        Returns:
            {
                'action': 'buy' | 'sell' | 'hold',
                'score': float,  # 综合得分
                'confidence': float,  # 置信度
                'position_ratio': float,  # 建议仓位比例
                'reason': str,  # 原因描述
            }
        """
        pass

    @abstractmethod
    def get_param_space(self) -> Dict[str, Dict]:
        """定义 Optuna 参数空间

        Returns:
            {参数名: {'type': 'float'|'int', 'low': ..., 'high': ..., 'step': ...}}
        """
        pass

    def get_default_params(self, regime: str = "neutral") -> Dict[str, Any]:
        """获取默认参数（当 Optuna 优化失败时的回退方案）"""
        space = self.get_param_space()
        defaults = {}
        for key, spec in space.items():
            if spec["type"] == "float":
                defaults[key] = (spec["low"] + spec["high"]) / 2
            elif spec["type"] == "int":
                defaults[key] = int((spec["low"] + spec["high"]) / 2)
        return defaults

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """校验参数合法性"""
        space = self.get_param_space()
        for key, spec in space.items():
            if key not in params:
                return False
            val = params[key]
            if spec["type"] == "float":
                if not (spec["low"] <= val <= spec["high"]):
                    return False
            elif spec["type"] == "int":
                if not (spec["low"] <= int(val) <= spec["high"]):
                    return False
        return True

    # ===== 向量化接口（步骤1：扩展基类，新增） =====
    def generate_signals_vectorized(
        self,
        df: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        regime: str = "neutral",
    ) -> pd.DataFrame:
        """
        向量化信号生成（默认实现：回退到逐行调用 generate_signal）。
        子类可覆写此方法以实现真正的向量化计算。

        输入/输出契约（必须遵守）：
        - 输入 df：至少包含时间索引和 OHLCV；因子列由具体策略约定（如 Combined_Score、atr 等）。
        - 输出 df：在输入 df 的副本上新增以下列（如果列已存在则覆盖）：
            - signal_action: int（1=买, -1=卖, 0=持有）
            - signal_score: float（综合得分）
            - signal_confidence: float（置信度）
            - target_position_ratio: float（建议仓位比例，0.0~1.0）
            - signal_reason: str（原因描述，可短文本）
        - 列名固定，方便后续向量化引擎/风控统一使用。

        注意：本基类提供“安全的逐行回退”实现，不改动原 df，返回 df.copy()。
        """

        if params is None:
            params = self.get_default_params(regime)

        # 创建副本避免副作用
        out = df.copy()

        # 预分配列，保证类型一致
        n = len(out)
        out["signal_action"] = np.zeros(n, dtype=np.int8)
        out["signal_score"] = np.full(n, 0.5, dtype=np.float64)
        out["signal_confidence"] = np.full(n, 0.0, dtype=np.float64)
        out["target_position_ratio"] = np.zeros(n, dtype=np.float64)
        out["signal_reason"] = ""

        # 逐行调用 generate_signal，并映射到向量化列
        action_map = {"buy": 1, "sell": -1, "hold": 0}
        for idx in out.index:
            sig = self.generate_signal(df, idx, params, regime)

            act_str = sig.get("action", "hold")
            act_int = action_map.get(act_str, 0)
            score = float(sig.get("score", 0.5))
            confidence = float(sig.get("confidence", 0.0))
            pos_ratio = float(sig.get("position_ratio", 0.0))
            reason = str(sig.get("reason", ""))

            out.at[idx, "signal_action"] = act_int
            out.at[idx, "signal_score"] = score
            out.at[idx, "signal_confidence"] = confidence
            out.at[idx, "target_position_ratio"] = pos_ratio
            out.at[idx, "signal_reason"] = reason

        return out

# -*- coding: utf-8 -*-
"""
复合信号策略
基于多因子加权得分 + Transformer置信度过滤的复合信号策略
从原始 backtest_market_v2.py 的交易逻辑迁移而来
"""
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from data.types import get_limit_ratio


class CompoundSignalStrategy(BaseStrategy):
    """复合信号策略

    信号生成逻辑：
    1. 综合得分超过买入阈值 -> 生成买入信号
    2. Transformer置信度过滤 -> 低置信度信号被拦截
    3. ATR动态仓位 -> 根据波动率调整仓位大小
    4. 移动止损/止盈/时间止损 -> 生成卖出信号
    """
    name = "compound_signal"
    description = "多因子加权 + Transformer置信度 + ATR动态仓位的复合策略"
    keep = True

    # ===== 保留原有单步接口（与回测/实盘兼容） =====
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        params: Dict[str, Any],
        regime: str = "neutral",
    ) -> Dict[str, Any]:
        """生成交易信号"""
        if idx < 60:
            return {
                "action": "hold",
                "score": 0.5,
                "confidence": 0.0,
                "position_ratio": 0.0,
                "reason": "数据不足",
            }

        score = df["Combined_Score"].iloc[idx]
        price = df["Close"].iloc[idx]

        # Transformer 因子
        transformer_prob = (
            df["transformer_prob"].iloc[idx]
            if "transformer_prob" in df.columns
            else 0.5
        )
        transformer_conf = (
            df["transformer_conf"].iloc[idx]
            if "transformer_conf" in df.columns
            else 0.0
        )

        # ========== 买入信号 ==========
        buy_threshold = params.get("buy_threshold", 0.6)
        confidence_threshold = params.get("confidence_threshold", 0.5)
        transformer_buy_threshold = params.get("transformer_buy_threshold", 0.6)

        if score > buy_threshold:
            # 置信度过滤
            if transformer_conf < confidence_threshold:
                return {
                    "action": "hold",
                    "score": score,
                    "confidence": transformer_conf,
                    "position_ratio": 0.0,
                    "reason": f"置信度不足({transformer_conf:.2f}<{confidence_threshold})",
                }

            # AI概率过滤
            if transformer_prob < transformer_buy_threshold:
                return {
                    "action": "hold",
                    "score": score,
                    "confidence": transformer_conf,
                    "position_ratio": 0.0,
                    "reason": f"AI概率不足({transformer_prob:.2f}<{transformer_buy_threshold})",
                }

            # ★ 修复B6：弱势市场不再一刀切阻断，改为限制仓位
            if regime == "bear":
                # bear 市场只允许极强信号
                if score < 0.85:
                    return {
                        "action": "hold",
                        "score": score,
                        "confidence": transformer_conf,
                        "position_ratio": 0.0,
                        "reason": f"空头市场信号不足({score:.2f}<0.85)",
                    }
            elif regime == "weak":
                # 弱势市场降低仓位而非阻断
                return {
                    "action": "hold",
                    "score": score,
                    "confidence": transformer_conf,
                    "position_ratio": 0.3,
                    "reason": "弱势市场阻断",
                }
            # ATR动态仓位
            atr = df["atr"].iloc[idx] if "atr" in df.columns else price * 0.02
            if pd.isna(atr) or atr <= 0:
                atr = price * 0.02
            daily_vol = atr / price
            if daily_vol <= 0 or not np.isfinite(daily_vol):
                daily_vol = 0.02
            target_annual_vol = 0.10
            position_ratio = target_annual_vol / (daily_vol * np.sqrt(252) + 1e-6)
            position_ratio = min(max(position_ratio, 0.1), 1.0)

            # 信号强度调整
            if "transformer_pred_ret" in df.columns:
                pred_ret = df["transformer_pred_ret"].iloc[idx]
                signal_strength = max(0.5, min(1.5, 1 + pred_ret / 0.05))
                position_ratio *= signal_strength

            return {
                "action": "buy",
                "score": score,
                "confidence": transformer_conf,
                "position_ratio": position_ratio,
                "reason": f"买入(score={score:.2f}, pos={position_ratio:.2f})",
            }

        # ========== 卖出信号 ==========
        sell_threshold = params.get("sell_threshold", -0.2)

        # 综合得分跌破阈值 -> 卖出
        if score < sell_threshold:
            return {
                "action": "sell",
                "score": score,
                "confidence": transformer_conf,
                "position_ratio": 0.0,
                "reason": f"得分跌破阈值({score:.2f}<{sell_threshold:.2f})",
            }

        # AI概率卖出
        transformer_sell_threshold = params.get("transformer_sell_threshold", 0.3)
        if transformer_prob < transformer_sell_threshold:
            return {
                "action": "sell",
                "score": score,
                "confidence": transformer_conf,
                "position_ratio": 0.0,
                "reason": f"AI概率过低({transformer_prob:.2f}<{transformer_sell_threshold:.2f})",
            }

        # 止损/止盈/移动止损（需要持仓成本）
        if (
            "position_price" in df.columns
            and not pd.isna(df["position_price"].iloc[idx])
            and df["position_price"].iloc[idx] > 0
        ):
            cost_price = df["position_price"].iloc[idx]
            pnl_ratio = (price - cost_price) / cost_price if cost_price > 0 else 0.0

            stop_loss = params.get("stop_loss", -0.08)
            take_profit = params.get("take_profit", 0.20)

            # ★ M6 修复：与 engine.py 统一，使用 ATR 倍数止盈
            take_profit_multiplier = params.get("take_profit_multiplier", 3.0)

            # 固定止损
            if pnl_ratio <= stop_loss:
                return {
                    "action": "sell",
                    "score": score,
                    "confidence": transformer_conf,
                    "position_ratio": 0.0,
                    "reason": f"固定止损({pnl_ratio:.2%}<={stop_loss:.2%})",
                }

            # ATR 动态止盈
            if 'atr' in df.columns and not pd.isna(df['atr'].iloc[idx]):
                atr = df['atr'].iloc[idx]
                if pnl_ratio >= take_profit_multiplier * (atr / cost_price):
                    return {
                        "action": "sell",
                        "score": score,
                        "confidence": transformer_conf,
                        "position_ratio": 0.0,
                        "reason": f"动态止盈({pnl_ratio:.2%}>=ATR*{take_profit_multiplier})",
                    }

            # 移动止损
            tp1 = params.get("trailing_profit_level1", 0.06)
            tp2 = params.get("trailing_profit_level2", 0.12)
            td1 = params.get("trailing_drawdown_level1", 0.08)
            td2 = params.get("trailing_drawdown_level2", 0.04)

            # 追踪高点
            peak_ratio = (
                df["peak_ratio"].iloc[idx]
                if "peak_ratio" in df.columns
                else pnl_ratio
            )
            if tp1 <= 0 or tp2 <= 0 or td1 <= 0 or td2 <= 0:
                # 参数异常时跳过移动止损
                pass
            else:
                if peak_ratio >= tp2 and (peak_ratio - pnl_ratio) >= td2:
                    return {
                        "action": "sell",
                        "score": score,
                        "confidence": transformer_conf,
                        "position_ratio": 0.0,
                        "reason": f"移动止损2级(peak={peak_ratio:.2%}, drawdown={peak_ratio - pnl_ratio:.2%})",
                    }
                if peak_ratio >= tp1 and (peak_ratio - pnl_ratio) >= td1:
                    return {
                        "action": "sell",
                        "score": score,
                        "confidence": transformer_conf,
                        "position_ratio": 0.0,
                        "reason": f"移动止损1级(peak={peak_ratio:.2%}, drawdown={peak_ratio - pnl_ratio:.2%})",
                    }

            # 时间止损（需要持仓日期）
            if "hold_days" in df.columns and not pd.isna(df["hold_days"].iloc[idx]):
                hold_days_limit = params.get("hold_days", 15)
                if df["hold_days"].iloc[idx] >= hold_days_limit:
                    return {
                        "action": "sell",
                        "score": score,
                        "confidence": transformer_conf,
                        "position_ratio": 0.0,
                        "reason": f"时间止损(hold_days={df['hold_days'].iloc[idx]}>={hold_days_limit})",
                    }

        # 默认持有
        return {
            "action": "hold",
            "score": score,
            "confidence": transformer_conf,
            "position_ratio": 0.0,
            "reason": "持有",
        }

    # ===== 参数空间（保持原逻辑） =====
    def get_param_space(self) -> Dict[str, Dict]:
        """定义 Optuna 参数空间"""
        return {
            # 买入阈值
            "buy_threshold": {
                "type": "float",
                "low": 0.4,
                "high": 0.9,
                "step": 0.05,
            },
            # 卖出阈值（得分低于该值卖出）
            "sell_threshold": {
                "type": "float",
                "low": -0.5,
                "high": 0.1,
                "step": 0.05,
            },
            # Transformer 相关阈值
            "transformer_buy_threshold": {
                "type": "float",
                "low": 0.3,
                "high": 0.9,
                "step": 0.05,
            },
            "transformer_sell_threshold": {
                "type": "float",
                "low": 0.1,
                "high": 0.5,
                "step": 0.05,
            },
            "confidence_threshold": {
                "type": "float",
                "low": 0.1,
                "high": 0.9,
                "step": 0.05,
            },
            # 止损/止盈
            "stop_loss": {
                "type": "float",
                "low": -0.15,
                "high": -0.03,
                "step": 0.01,
            },
            # ★ M6 修复：与 engine.py 统一，使用 ATR 倍数
            "take_profit_multiplier": {
                "type": "float",
                "low": 1.5,
                "high": 5.0,
                "step": 0.25,
            },
            # 移动止损（触发阈值 + 回撤幅度）
            "trailing_profit_level1": {
                "type": "float",
                "low": 0.03,
                "high": 0.15,
                "step": 0.01,
            },
            "trailing_profit_level2": {
                "type": "float",
                "low": 0.08,
                "high": 0.25,
                "step": 0.01,
            },
            "trailing_drawdown_level1": {
                "type": "float",
                "low": 0.03,
                "high": 0.12,
                "step": 0.01,
            },
            "trailing_drawdown_level2": {
                "type": "float",
                "low": 0.02,
                "high": 0.08,
                "step": 0.01,
            },
            # 时间止损（持仓天数上限）
            "hold_days": {
                "type": "int",
                "low": 1,
                "high": 60,
                "step": 1,
            },
        }

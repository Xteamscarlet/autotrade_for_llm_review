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

            # 弱势市场阻断
            if regime == "weak":
                return {
                    "action": "hold",
                    "score": score,
                    "confidence": transformer_conf,
                    "position_ratio": 0.0,
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

            # 固定止损
            if pnl_ratio <= stop_loss:
                return {
                    "action": "sell",
                    "score": score,
                    "confidence": transformer_conf,
                    "position_ratio": 0.0,
                    "reason": f"固定止损({pnl_ratio:.2%}<={stop_loss:.2%})",
                }

            # 固定止盈
            if pnl_ratio >= take_profit:
                return {
                    "action": "sell",
                    "score": score,
                    "confidence": transformer_conf,
                    "position_ratio": 0.0,
                    "reason": f"固定止盈({pnl_ratio:.2%}>={take_profit:.2%})",
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
            "take_profit": {
                "type": "float",
                "low": 0.05,
                "high": 0.40,
                "step": 0.05,
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

    # ===== 步骤1：新增向量化实现 =====
    def generate_signals_vectorized(
        self,
        df: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        regime: str = "neutral",
    ) -> pd.DataFrame:
        """
        复合信号策略的向量化信号生成。
        与 generate_signal 保持逻辑一致，但全部使用列运算，提升回测性能。

        输入 df（建议包含的字段，尽量与原有逻辑一致）：
        - OHLCV: 'Open', 'High', 'Low', 'Close', 'Volume'
        - 因子: 'Combined_Score', 'transformer_prob', 'transformer_conf', 'transformer_pred_ret', 'atr'
        -（可选）持仓状态（由引擎维护）: 'position_price', 'peak_ratio', 'hold_days'

        输出 df（在输入 df 基础上新增）：
        - signal_action: 1(买), -1(卖), 0(持有)
        - signal_score: 综合得分
        - signal_confidence: 置信度
        - target_position_ratio: 建议仓位比例（买入时>0，卖出/持有时=0）
        - signal_reason: 原因描述（长度较长，用于调试/日志）
        """
        if params is None:
            params = self.get_default_params(regime)

        out = df.copy()

        # ---- 必备列检查与容错（保持与 generate_signal 一致） ----
        score_col = "Combined_Score"
        if score_col not in out.columns:
            raise KeyError(f"输入 df 必须包含 '{score_col}' 列")

        # 收盘价
        price = out["Close"].astype(np.float64).values
        score = out[score_col].astype(np.float64).values
        n = len(out)

        # Transformer 因子（缺失时用 0.5/0.0 填充）
        transformer_prob = (
            out["transformer_prob"].astype(np.float64).values
            if "transformer_prob" in out.columns
            else np.full(n, 0.5)
        )
        transformer_conf = (
            out["transformer_conf"].astype(np.float64).values
            if "transformer_conf" in out.columns
            else np.full(n, 0.0)
        )

        # ATR（缺失时用 price * 0.02 填充）
        atr = (
            out["atr"].astype(np.float64).values
            if "atr" in out.columns
            else np.full(n, np.nan)
        )
        atr_safe = np.where(
            (~np.isnan(atr)) & (atr > 0),
            atr,
            price * 0.02,
        )

        daily_vol = atr_safe / np.where(price > 0, price, 1e-8)
        daily_vol = np.where(
            (daily_vol > 0) & np.isfinite(daily_vol),
            daily_vol,
            0.02,
        )

        # 可选持仓列（缺失时全 nan，不触发卖出逻辑）
        position_price = (
            out["position_price"].astype(np.float64).values
            if "position_price" in out.columns
            else np.full(n, np.nan)
        )
        peak_ratio = (
            out["peak_ratio"].astype(np.float64).values
            if "peak_ratio" in out.columns
            else np.full(n, np.nan)
        )
        hold_days = (
            out["hold_days"].astype(np.float64).values
            if "hold_days" in out.columns
            else np.full(n, np.nan)
        )

        # ---- 参数提取 ----
        buy_threshold = float(params.get("buy_threshold", 0.6))
        sell_threshold = float(params.get("sell_threshold", -0.2))
        transformer_buy_threshold = float(params.get("transformer_buy_threshold", 0.6))
        transformer_sell_threshold = float(params.get("transformer_sell_threshold", 0.3))
        confidence_threshold = float(params.get("confidence_threshold", 0.5))
        stop_loss = float(params.get("stop_loss", -0.08))
        take_profit = float(params.get("take_profit", 0.20))
        tp1 = float(params.get("trailing_profit_level1", 0.06))
        tp2 = float(params.get("trailing_profit_level2", 0.12))
        td1 = float(params.get("trailing_drawdown_level1", 0.08))
        td2 = float(params.get("trailing_drawdown_level2", 0.04))
        hold_days_limit = int(params.get("hold_days", 15))
        target_annual_vol = 0.10

        # ---- 预分配输出 ----
        signal_action = np.zeros(n, dtype=np.int8)  # 1/-1/0
        target_position_ratio = np.zeros(n, dtype=np.float64)  # 买入时>0，卖出/持有=0
        signal_score = score.copy()
        signal_confidence = transformer_conf.copy()
        signal_reason = np.empty(n, dtype=object)  # str 数组

        # 默认 reason
        for i in range(n):
            signal_reason[i] = "持有"

        # ---- 数据不足前 60 根K线：全部 hold ----
        insufficient = np.arange(n) < 60
        signal_action[insufficient] = 0
        target_position_ratio[insufficient] = 0.0
        for i in np.where(insufficient)[0]:
            signal_reason[i] = "数据不足"

        # ---- 买入候选掩码 ----
        buy_candidate = score > buy_threshold  # 首先满足得分阈值

        # 置信度过滤
        confidence_filter = transformer_conf < confidence_threshold
        buy_blocked_by_confidence = buy_candidate & confidence_filter
        signal_action[buy_blocked_by_confidence] = 0
        target_position_ratio[buy_blocked_by_confidence] = 0.0
        for i in np.where(buy_blocked_by_confidence)[0]:
            signal_reason[i] = (
                f"置信度不足({transformer_conf[i]:.2f}<{confidence_threshold:.2f})"
            )
        buy_candidate = buy_candidate & (~confidence_filter)

        # AI概率过滤
        prob_filter = transformer_prob < transformer_buy_threshold
        buy_blocked_by_prob = buy_candidate & prob_filter
        signal_action[buy_blocked_by_prob] = 0
        target_position_ratio[buy_blocked_by_prob] = 0.0
        for i in np.where(buy_blocked_by_prob)[0]:
            signal_reason[i] = (
                f"AI概率不足({transformer_prob[i]:.2f}<{transformer_buy_threshold:.2f})"
            )
        buy_candidate = buy_candidate & (~prob_filter)

        # 弱势市场阻断（regime='weak'时全局阻断买入）
        if regime == "weak":
            buy_blocked_by_regime = buy_candidate
            signal_action[buy_blocked_by_regime] = 0
            target_position_ratio[buy_blocked_by_regime] = 0.0
            for i in np.where(buy_blocked_by_regime)[0]:
                signal_reason[i] = "弱势市场阻断"
            buy_candidate = buy_candidate & False  # 清空

        # ---- ATR动态仓位（仅对最终买入候选计算） ----
        vol_position = target_annual_vol / (daily_vol * np.sqrt(252) + 1e-6)
        vol_position = np.clip(vol_position, 0.1, 1.0)

        # 信号强度调整（如果存在 'transformer_pred_ret'）
        signal_strength = np.ones(n)
        if "transformer_pred_ret" in out.columns:
            pred_ret = out["transformer_pred_ret"].astype(np.float64).values
            signal_strength = np.clip(1.0 + pred_ret / 0.05, 0.5, 1.5)

        final_position_ratio = vol_position * signal_strength

        # 对买入候选写回
        target_position_ratio[buy_candidate] = final_position_ratio[buy_candidate]
        signal_action[buy_candidate] = 1
        for i in np.where(buy_candidate)[0]:
            signal_reason[i] = f"买入(score={score[i]:.2f}, pos={final_position_ratio[i]:.2f})"

        # ---- 卖出逻辑（以下在“非买入候选”的行上判定，避免覆盖买入） ----
        non_buy = ~buy_candidate & (~insufficient)

        # 综合得分跌破阈值
        sell_by_score = non_buy & (score < sell_threshold)
        signal_action[sell_by_score] = -1
        target_position_ratio[sell_by_score] = 0.0
        for i in np.where(sell_by_score)[0]:
            signal_reason[i] = f"得分跌破阈值({score[i]:.2f}<{sell_threshold:.2f})"

        # AI概率过低卖出
        sell_by_ai_prob = non_buy & (transformer_prob < transformer_sell_threshold)
        signal_action[sell_by_ai_prob] = -1
        target_position_ratio[sell_by_ai_prob] = 0.0
        for i in np.where(sell_by_ai_prob)[0]:
            signal_reason[i] = (
                f"AI概率过低({transformer_prob[i]:.2f}<{transformer_sell_threshold:.2f})"
            )

        # 持仓状态相关卖出（固定止损/止盈/移动止损/时间止损）
        has_position = (
            (~np.isnan(position_price))
            & (position_price > 0)
            & non_buy
        )
        if np.any(has_position):
            pnl_ratio = np.where(
                has_position & (position_price > 0),
                (price - position_price) / position_price,
                0.0,
            )

            # 固定止损
            sell_by_stop_loss = has_position & (pnl_ratio <= stop_loss)
            signal_action[sell_by_stop_loss] = -1
            target_position_ratio[sell_by_stop_loss] = 0.0
            for i in np.where(sell_by_stop_loss)[0]:
                signal_reason[i] = f"固定止损({pnl_ratio[i]:.2%}<={stop_loss:.2%})"

            # 固定止盈
            sell_by_take_profit = has_position & (pnl_ratio >= take_profit)
            signal_action[sell_by_take_profit] = -1
            target_position_ratio[sell_by_take_profit] = 0.0
            for i in np.where(sell_by_take_profit)[0]:
                signal_reason[i] = f"固定止盈({pnl_ratio[i]:.2%}>={take_profit:.2%})"

            # 移动止损（需要 peak_ratio）
            peak_safe = np.where(
                (~np.isnan(peak_ratio)),
                peak_ratio,
                pnl_ratio,
            )
            param_ok = (tp1 > 0) & (tp2 > 0) & (td1 > 0) & (td2 > 0)
            if param_ok:
                # 二级移动止损
                sell_by_trailing2 = (
                    has_position
                    & (peak_safe >= tp2)
                    & ((peak_safe - pnl_ratio) >= td2)
                )
                signal_action[sell_by_trailing2] = -1
                target_position_ratio[sell_by_trailing2] = 0.0
                for i in np.where(sell_by_trailing2)[0]:
                    signal_reason[i] = (
                        f"移动止损2级(peak={peak_safe[i]:.2%}, "
                        f"drawdown={peak_safe[i] - pnl_ratio[i]:.2%})"
                    )

                # 一级移动止损
                sell_by_trailing1 = (
                    has_position
                    & (peak_safe >= tp1)
                    & ((peak_safe - pnl_ratio) >= td1)
                    & (~sell_by_trailing2)  # 不与二级重复
                )
                signal_action[sell_by_trailing1] = -1
                target_position_ratio[sell_by_trailing1] = 0.0
                for i in np.where(sell_by_trailing1)[0]:
                    signal_reason[i] = (
                        f"移动止损1级(peak={peak_safe[i]:.2%}, "
                        f"drawdown={peak_safe[i] - pnl_ratio[i]:.2%})"
                    )

            # 时间止损
            sell_by_time = (
                has_position
                & (~np.isnan(hold_days))
                & (hold_days >= hold_days_limit)
            )
            signal_action[sell_by_time] = -1
            target_position_ratio[sell_by_time] = 0.0
            for i in np.where(sell_by_time)[0]:
                signal_reason[i] = (
                    f"时间止损(hold_days={hold_days[i]:.0f}>={hold_days_limit})"
                )

        # ---- 写回 out（不覆盖原有因子列） ----
        out["signal_action"] = signal_action
        out["signal_score"] = signal_score
        out["signal_confidence"] = signal_confidence
        out["target_position_ratio"] = target_position_ratio
        out["signal_reason"] = signal_reason

        return out

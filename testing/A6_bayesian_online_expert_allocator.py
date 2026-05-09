# -*- coding: utf-8 -*-
"""
A6 D-ONS OBS expert allocator.

This module implements the strategy-level online expert ensemble described in
``etf_rank_project_plan.md``.  It intentionally keeps A6 local to this file:
optional A0/A1/A5 imports are used when available, while small fallbacks keep the
module runnable in isolation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from A0_equal_weight import (  # type: ignore
        build_trade_rows as _a0_build_trade_rows,
        compute_performance_metrics as _a0_compute_performance_metrics,
        default_kofr_path as _a0_default_kofr_path,
        default_regime_path as _a0_default_regime_path,
        get_valid_universe as _a0_get_valid_universe,
        kofr_cash_return_for_date as _a0_kofr_cash_return_for_date,
        load_kofr_table as _a0_load_kofr_table,
        load_regime_table as _a0_load_regime_table,
        load_rank_panel as _a0_load_rank_panel,
        normalize_sleeve_weights as _a0_normalize_sleeve_weights,
        regime_info_for_date as _a0_regime_info_for_date,
        regime_stock_ratio_for_date as _a0_regime_stock_ratio_for_date,
        rebalance_and_mark_to_market_next_day as _a0_rebalance_next_day,
        validate_backtest_panel as _a0_validate_backtest_panel,
    )
except Exception:  # pragma: no cover - optional project module
    _a0_build_trade_rows = None
    _a0_compute_performance_metrics = None
    _a0_default_kofr_path = None
    _a0_default_regime_path = None
    _a0_get_valid_universe = None
    _a0_kofr_cash_return_for_date = None
    _a0_load_kofr_table = None
    _a0_load_regime_table = None
    _a0_load_rank_panel = None
    _a0_normalize_sleeve_weights = None
    _a0_regime_info_for_date = None
    _a0_regime_stock_ratio_for_date = None
    _a0_rebalance_next_day = None
    _a0_validate_backtest_panel = None

try:
    from A1_rule_based_rank_allocator import (  # type: ignore
        compute_composite_score as _a1_compute_composite_score,
        generate_rule_based_weight as _a1_generate_rule_based_weight,
    )
except Exception:  # pragma: no cover - optional project module
    _a1_compute_composite_score = None
    _a1_generate_rule_based_weight = None

try:
    from A5_bayesian_winner_loser_allocator import (  # type: ignore
        generate_A5_weight as _a5_generate_A5_weight,
    )
except Exception:  # pragma: no cover - optional project module
    _a5_generate_A5_weight = None


INITIAL_NAV = 1_000_000_000.0
COMMISSION_RATE = 0.00015
STOCK_RATIO = 1.0
TOP_K_VALUES = (3, 5, 7, 10)
EPS = 1e-12


CORE_EXPERTS = (
    "A0_EQUAL_WEIGHT",
    "MARKET_ONLY",
    "FLOW_ONLY",
    "ROTATION_ONLY",
    "A1_RULE_COMPOSITE",
    "A5_BAYESIAN_WINNER",
)

FULL_EXPERTS = (
    "MARKET_ONLY",
    "FLOW_ONLY",
    "ROTATION_ONLY",
    "A0_EQUAL_WEIGHT",
    "A1_RULE_COMPOSITE",
    "A2a_LLM_OPAQUE",
    "A2b_LLM_SEMANTIC",
    "A3_LLM_POLICY",
    "A4_RULE_LLM_BLEND",
    "A5_BAYESIAN_WINNER",
)

FULL_EXPERT_DIRS = {
    "A0_EQUAL_WEIGHT": "A0",
    "A1_RULE_COMPOSITE": "A1_k{top_k}",
    "A2a_LLM_OPAQUE": "A2a_k{top_k}",
    "A2b_LLM_SEMANTIC": "A2b_k{top_k}",
    "A3_LLM_POLICY": "A3_k{top_k}",
    "A4_RULE_LLM_BLEND": "A4_k{top_k}",
    "A5_BAYESIAN_WINNER": "A5_k{top_k}",
}


@dataclass
class BacktestConfig:
    initial_nav: float = INITIAL_NAV
    commission_rate: float = COMMISSION_RATE
    stock_ratio: float = STOCK_RATIO
    cash_return: float = 0.0
    commission_fixed_point_tol: float = 1e-10
    commission_fixed_point_max_iter: int = 50


@dataclass
class PortfolioState:
    nav: float
    cash: float
    quantities: Dict[str, float]


@dataclass
class DONSState:
    expert_names: List[str]
    pi: np.ndarray
    P: np.ndarray
    P0: np.ndarray
    eta: float
    gamma: float


def load_rank_panel(path: str | Path) -> pd.DataFrame:
    if _a0_load_rank_panel is not None:
        return _a0_load_rank_panel(path)
    panel = pd.read_csv(path)
    if "date" not in panel.columns:
        raise ValueError("rank_panel.csv must contain a date column")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    if "ETF_id" not in panel.columns:
        panel["ETF_id"] = panel["ticker"]
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def validate_backtest_panel(panel: pd.DataFrame) -> None:
    if _a0_validate_backtest_panel is not None:
        _a0_validate_backtest_panel(panel)
        return
    required = {"date", "ticker", "open", "close", "market_score", "flow_score", "rotation_score"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"rank panel missing required columns: {sorted(missing)}")
    if panel.duplicated(["date", "ticker"]).any():
        raise ValueError("rank panel contains duplicate date/ticker rows")
    if (panel["open"] <= 0).any() or (panel["close"] <= 0).any():
        raise ValueError("open and close must be positive")


def get_valid_universe(day_df: pd.DataFrame) -> pd.DataFrame:
    if _a0_get_valid_universe is not None:
        return _a0_get_valid_universe(day_df).copy()
    mask = pd.Series(True, index=day_df.index)
    if "tradable" in day_df.columns:
        mask &= day_df["tradable"].fillna(False).astype(bool)
    if "all_rank_valid" in day_df.columns:
        mask &= day_df["all_rank_valid"].fillna(False).astype(bool)
    for col in ("market_score", "flow_score", "rotation_score"):
        mask &= day_df[col].notna()
    mask &= (day_df["open"] > 0) & (day_df["close"] > 0)
    return day_df.loc[mask].copy()


def normalize_sleeve_weights(raw_weights: Mapping[str, float], valid_tickers: Iterable[str]) -> Dict[str, float]:
    valid = [str(t) for t in valid_tickers]
    if _a0_normalize_sleeve_weights is not None:
        raw_series = pd.Series(raw_weights, dtype=float)
        weights, _fallback_used = _a0_normalize_sleeve_weights(raw_series, valid)
        return dict(weights)
    cleaned = {ticker: max(0.0, float(raw_weights.get(ticker, 0.0))) for ticker in valid}
    total = sum(cleaned.values())
    if total <= EPS:
        if not valid:
            return {}
        return {ticker: 1.0 / len(valid) for ticker in valid}
    return {ticker: weight / total for ticker, weight in cleaned.items()}


def coerce_weight_result(result: object) -> Dict[str, float]:
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, pd.Series):
        return {str(ticker): float(weight) for ticker, weight in result.items()}
    if isinstance(result, Mapping):
        return {str(ticker): float(weight) for ticker, weight in result.items()}
    raise TypeError(f"unsupported weight result type: {type(result)!r}")


def compute_composite_score(day_df: pd.DataFrame) -> pd.Series:
    if _a1_compute_composite_score is not None:
        return _a1_compute_composite_score(day_df)
    return (
        0.30 * day_df["market_score"].fillna(0.0).clip(lower=0.0)
        + 0.35 * day_df["flow_score"].fillna(0.0).clip(lower=0.0)
        + 0.35 * day_df["rotation_score"].fillna(0.0).clip(lower=0.0)
    )


def apply_top_k_filter(weights: Mapping[str, float], top_k: int) -> Dict[str, float]:
    if top_k >= 10:
        return dict(weights)
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    keep = {ticker for ticker, _ in ordered[:top_k]}
    return {ticker: weight for ticker, weight in weights.items() if ticker in keep}


def score_proportional_weights(day_df: pd.DataFrame, score: pd.Series, top_k: int) -> Dict[str, float]:
    valid = get_valid_universe(day_df)
    if valid.empty:
        return {}
    score = score.reindex(valid.index).fillna(0.0).clip(lower=0.0)
    raw = dict(zip(valid["ticker"].astype(str), score.astype(float)))
    raw = apply_top_k_filter(raw, top_k)
    return normalize_sleeve_weights(raw, raw.keys())


def make_equal_weight_expert(day_df: pd.DataFrame) -> Dict[str, float]:
    valid = get_valid_universe(day_df)
    return normalize_sleeve_weights({}, valid["ticker"].astype(str))


def make_single_rank_expert(day_df: pd.DataFrame, score_col: str, top_k: int) -> Dict[str, float]:
    return score_proportional_weights(day_df, day_df[score_col], top_k)


def make_composite_expert(day_df: pd.DataFrame, top_k: int) -> Dict[str, float]:
    if _a1_generate_rule_based_weight is not None:
        try:
            return coerce_weight_result(_a1_generate_rule_based_weight(day_df, top_k=top_k))
        except TypeError:
            return coerce_weight_result(_a1_generate_rule_based_weight(day_df))
    return score_proportional_weights(day_df, compute_composite_score(day_df), top_k)


def _bucketize_score(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value >= 0.67:
        return "top"
    if value >= 0.33:
        return "middle"
    return "bottom"


def _panel_with_a5_labels(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["ticker", "date"]).copy()
    df["etf_next_cc_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1.0
    if "index_close" in df.columns:
        index_by_date = df[["date", "index_close"]].drop_duplicates("date").sort_values("date")
        index_by_date["index_next_cc_ret"] = index_by_date["index_close"].shift(-1) / index_by_date["index_close"] - 1.0
        df = df.drop(columns=["index_next_cc_ret"], errors="ignore").merge(
            index_by_date[["date", "index_next_cc_ret"]], on="date", how="left"
        )
    else:
        df["index_next_cc_ret"] = 0.0
    df["winner_label"] = df["etf_next_cc_ret"] > df["index_next_cc_ret"]
    for col, prefix in (("market_score", "M"), ("flow_score", "F"), ("rotation_score", "R")):
        df[f"{prefix}_bucket"] = df[col].map(_bucketize_score)
    df["state_key"] = (
        "M_"
        + df["M_bucket"]
        + "__F_"
        + df["F_bucket"]
        + "__R_"
        + df["R_bucket"]
    )
    return df


def make_A5_expert(day_df: pd.DataFrame, history_df: pd.DataFrame, top_k: int) -> Dict[str, float]:
    if _a5_generate_A5_weight is not None and "benchmark_beat_label" in history_df.columns:
        try:
            return coerce_weight_result(_a5_generate_A5_weight(day_df, history_df, top_k=top_k))
        except TypeError:
            return coerce_weight_result(_a5_generate_A5_weight(day_df, history_df))

    valid = get_valid_universe(day_df)
    if valid.empty:
        return {}

    alpha0, beta0, shrink_k = 1.0, 1.0, 20.0
    hist = history_df.dropna(subset=["winner_label", "state_key"]).copy()
    global_wins = float(hist["winner_label"].sum())
    global_n = float(len(hist))
    global_p = (alpha0 + global_wins) / (alpha0 + beta0 + global_n)

    grouped = hist.groupby("state_key")["winner_label"].agg(["sum", "count"])
    scores = {}
    for _, row in valid.iterrows():
        state_key = (
            f"M_{_bucketize_score(row['market_score'])}"
            f"__F_{_bucketize_score(row['flow_score'])}"
            f"__R_{_bucketize_score(row['rotation_score'])}"
        )
        if state_key in grouped.index:
            wins = float(grouped.loc[state_key, "sum"])
            n = float(grouped.loc[state_key, "count"])
            p_state = (alpha0 + wins) / (alpha0 + beta0 + n)
            lam = n / (n + shrink_k)
            score = lam * p_state + (1.0 - lam) * global_p
        else:
            score = global_p
        scores[str(row["ticker"])] = score

    scores = apply_top_k_filter(scores, top_k)
    return normalize_sleeve_weights(scores, scores.keys())


def make_core_expert_weights(day_df: pd.DataFrame, history_df: pd.DataFrame, top_k: int) -> Dict[str, Dict[str, float]]:
    return {
        "A0_EQUAL_WEIGHT": make_equal_weight_expert(day_df),
        "MARKET_ONLY": make_single_rank_expert(day_df, "market_score", top_k),
        "FLOW_ONLY": make_single_rank_expert(day_df, "flow_score", top_k),
        "ROTATION_ONLY": make_single_rank_expert(day_df, "rotation_score", top_k),
        "A1_RULE_COMPOSITE": make_composite_expert(day_df, top_k),
        "A5_BAYESIAN_WINNER": make_A5_expert(day_df, history_df, top_k),
    }


def initialize_dons_state(expert_names: Sequence[str], eta: float = 1.0, gamma: float = 0.99) -> DONSState:
    names = list(expert_names)
    n = len(names)
    if n == 0:
        raise ValueError("D-ONS requires at least one expert")
    P0 = np.eye(n, dtype=float)
    return DONSState(
        expert_names=names,
        pi=np.full(n, 1.0 / n, dtype=float),
        P=P0.copy(),
        P0=P0,
        eta=float(eta),
        gamma=float(gamma),
    )


def compute_expert_mixture_probs(dons_state: DONSState) -> Dict[str, float]:
    return dict(zip(dons_state.expert_names, dons_state.pi.astype(float)))


def project_to_simplex_p_metric(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project ``y`` onto the probability simplex in the P metric."""
    y = np.asarray(y, dtype=float)
    P = np.asarray(P, dtype=float)
    n = len(y)
    active = np.ones(n, dtype=bool)

    for _ in range(n + 1):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            return np.full(n, 1.0 / n)
        inactive_idx = np.flatnonzero(~active)
        Paa = P[np.ix_(idx, idx)]
        ya = y[idx]
        correction = np.zeros(len(idx))
        if len(inactive_idx) > 0:
            Pab = P[np.ix_(idx, inactive_idx)]
            correction = np.linalg.solve(Paa, Pab @ y[inactive_idx])
        inv_ones = np.linalg.solve(Paa, np.ones(len(idx)))
        base = ya + correction
        lam = (base.sum() - 1.0) / inv_ones.sum()
        xa = base - lam * inv_ones
        if np.all(xa > -1e-12):
            x = np.zeros(n, dtype=float)
            x[idx] = np.maximum(xa, 0.0)
            total = x.sum()
            if total <= EPS:
                return np.full(n, 1.0 / n)
            return x / total
        active[idx[xa <= 0.0]] = False

    x = np.maximum(y, 0.0)
    if x.sum() <= EPS:
        return np.full(n, 1.0 / n)
    return x / x.sum()


def update_dons_state(dons_state: DONSState, expert_net_gross_returns: Mapping[str, float]) -> DONSState:
    r = np.array(
        [max(EPS, float(expert_net_gross_returns.get(name, 1.0))) for name in dons_state.expert_names],
        dtype=float,
    )
    denom = max(EPS, float(dons_state.pi @ r))
    gradient = -r / denom
    hessian = np.outer(r, r) / (denom * denom)
    P_next = (1.0 - dons_state.gamma) * dons_state.P0 + dons_state.gamma * dons_state.P + hessian
    step = np.linalg.solve(P_next, gradient) / max(EPS, dons_state.eta)
    pi_tilde = dons_state.pi - step
    pi_next = project_to_simplex_p_metric(pi_tilde, P_next)
    return DONSState(
        expert_names=dons_state.expert_names,
        pi=pi_next,
        P=P_next,
        P0=dons_state.P0,
        eta=dons_state.eta,
        gamma=dons_state.gamma,
    )


def blend_expert_weights(
    expert_weights: Mapping[str, Mapping[str, float]],
    expert_probs: Mapping[str, float],
    valid_tickers: Iterable[str],
) -> Dict[str, float]:
    blended = {str(ticker): 0.0 for ticker in valid_tickers}
    for expert_name, prob in expert_probs.items():
        for ticker, weight in expert_weights.get(expert_name, {}).items():
            ticker = str(ticker)
            if ticker in blended:
                blended[ticker] += float(prob) * float(weight)
    return normalize_sleeve_weights(blended, blended.keys())


def initialize_portfolio_state(initial_nav: float = INITIAL_NAV) -> PortfolioState:
    return PortfolioState(nav=float(initial_nav), cash=float(initial_nav), quantities={})


def initialize_shadow_expert_states(expert_names: Sequence[str], initial_nav: float = INITIAL_NAV) -> Dict[str, PortfolioState]:
    return {name: initialize_portfolio_state(initial_nav) for name in expert_names}


def default_regime_path() -> Optional[Path]:
    if _a0_default_regime_path is None:
        return None
    return _a0_default_regime_path()


def load_regime_table(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if _a0_load_regime_table is None:
        return None
    return _a0_load_regime_table(path)


def default_kofr_path() -> Optional[Path]:
    if _a0_default_kofr_path is None:
        return None
    return _a0_default_kofr_path()


def load_kofr_table(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if _a0_load_kofr_table is None:
        return None
    return _a0_load_kofr_table(path)


def regime_stock_ratio_for_date(regime: Optional[pd.DataFrame], date: pd.Timestamp, default: float) -> float:
    if _a0_regime_stock_ratio_for_date is None:
        return float(default)
    return float(_a0_regime_stock_ratio_for_date(regime, date, default))


def kofr_cash_return_for_date(kofr: Optional[pd.DataFrame], date: pd.Timestamp, default: float = 0.0) -> float:
    if _a0_kofr_cash_return_for_date is None:
        return float(default)
    return float(_a0_kofr_cash_return_for_date(kofr, date, default))


def _display_path(path: Optional[Path]) -> Optional[str]:
    return path.name if path else None


def _price_map(day_df: pd.DataFrame, field: str) -> Dict[str, float]:
    return dict(zip(day_df["ticker"].astype(str), day_df[field].astype(float)))


def _safe_sum_values(quantities: Mapping[str, float], prices: Mapping[str, float]) -> float:
    return sum(float(qty) * float(prices.get(ticker, 0.0)) for ticker, qty in quantities.items())


def rebalance_and_mark_to_market_next_day(
    state: PortfolioState,
    target_sleeve_weights: Mapping[str, float],
    price_t: pd.DataFrame,
    price_t1: pd.DataFrame,
    config: BacktestConfig,
) -> Tuple[PortfolioState, Dict[str, float]]:
    close_t = _price_map(price_t, "close")
    open_t1 = _price_map(price_t1, "open")
    close_t1 = _price_map(price_t1, "close")

    old_close_value = _safe_sum_values(state.quantities, close_t)
    old_open_value = _safe_sum_values(state.quantities, open_t1)
    overnight_pnl = old_open_value - old_close_value
    nav_open_pre = state.cash + old_open_value
    old_values_open = {ticker: qty * open_t1.get(ticker, 0.0) for ticker, qty in state.quantities.items()}

    weight_sum = sum(max(0.0, float(weight)) for weight in target_sleeve_weights.values())
    normalized_weights = {
        str(ticker): max(0.0, float(weight)) / weight_sum
        for ticker, weight in target_sleeve_weights.items()
        if weight_sum > EPS
    }
    nav_open_post = nav_open_pre
    commission = 0.0
    target_values: Dict[str, float] = {}
    target_quantities: Dict[str, float] = {}
    for _ in range(config.commission_fixed_point_max_iter):
        target_quantities = {}
        target_values = {}
        for ticker, weight in normalized_weights.items():
            if ticker not in open_t1 or open_t1[ticker] <= 0:
                continue
            target_budget = config.stock_ratio * nav_open_post * weight
            qty = math.floor(target_budget / open_t1[ticker]) if target_budget > 0 else 0
            if qty <= 0:
                continue
            target_quantities[ticker] = float(qty)
            target_values[ticker] = float(qty) * open_t1[ticker]
        all_tickers = set(target_values).union(old_values_open)
        turnover_value = sum(abs(target_values.get(ticker, 0.0) - old_values_open.get(ticker, 0.0)) for ticker in all_tickers)
        next_commission = config.commission_rate * turnover_value
        next_nav_open_post = nav_open_pre - next_commission
        if abs(next_nav_open_post - nav_open_post) <= config.commission_fixed_point_tol:
            commission = next_commission
            nav_open_post = next_nav_open_post
            break
        commission = next_commission
        nav_open_post = next_nav_open_post

    quantities = target_quantities
    cash_open_post = nav_open_post - sum(target_values.values())
    cash = cash_open_post * (1.0 + config.cash_return)
    open_to_close_pnl = sum(qty * (close_t1.get(ticker, 0.0) - open_t1.get(ticker, 0.0)) for ticker, qty in quantities.items())
    nav_close = cash + _safe_sum_values(quantities, close_t1)
    turnover_ratio = turnover_value / nav_open_pre if nav_open_pre > EPS else 0.0
    daily_return = nav_close / state.nav - 1.0 if state.nav > EPS else 0.0

    next_state = PortfolioState(nav=nav_close, cash=cash, quantities=quantities)
    details = {
        "nav_open_pre": nav_open_pre,
        "nav_open_post": nav_open_post,
        "nav_close": nav_close,
        "overnight_pnl": overnight_pnl,
        "open_to_close_pnl": open_to_close_pnl,
        "commission": commission,
        "turnover_value": turnover_value,
        "turnover_ratio": turnover_ratio,
        "daily_return": daily_return,
        "cash_return": config.cash_return,
    }
    return next_state, details


def rebalance_and_mark_to_market_expert_shadow_state(
    expert_state: PortfolioState,
    expert_target_weights: Mapping[str, float],
    price_t: pd.DataFrame,
    price_t1: pd.DataFrame,
    config: BacktestConfig,
) -> Tuple[PortfolioState, float]:
    previous_nav = expert_state.nav
    next_state, _ = rebalance_and_mark_to_market_next_day(
        expert_state, expert_target_weights, price_t, price_t1, config
    )
    return next_state, max(EPS, next_state.nav / previous_nav) if previous_nav > EPS else 1.0


def _date_key(value: object) -> pd.Timestamp:
    return pd.Timestamp(value)


def _next_date_map(dates: Sequence[pd.Timestamp]) -> Dict[pd.Timestamp, pd.Timestamp]:
    return {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}


def _index_gross_return(price_t: pd.DataFrame, price_t1: pd.DataFrame) -> float:
    if "index_close" not in price_t.columns or "index_close" not in price_t1.columns:
        return 1.0
    index_t = float(price_t["index_close"].dropna().iloc[0])
    index_t1 = float(price_t1["index_close"].dropna().iloc[0])
    if index_t <= 0 or index_t1 <= 0:
        return 1.0
    return index_t1 / index_t


def _effective_num_positions(weights: Mapping[str, float]) -> float:
    vals = np.array(list(weights.values()), dtype=float)
    denom = float(np.square(vals).sum())
    return 1.0 / denom if denom > EPS else 0.0


def _entropy(probs: Mapping[str, float]) -> float:
    vals = np.array([p for p in probs.values() if p > EPS], dtype=float)
    return float(-(vals * np.log(vals)).sum()) if len(vals) else 0.0


def compute_performance_metrics(daily_results: pd.DataFrame, initial_nav: float = INITIAL_NAV) -> Dict[str, float]:
    if _a0_compute_performance_metrics is not None:
        return dict(_a0_compute_performance_metrics(daily_results))
    if daily_results.empty:
        return {}
    returns = daily_results["daily_return"].astype(float)
    nav = daily_results["nav_close"].astype(float)
    total_return = nav.iloc[-1] / initial_nav - 1.0
    periods = max(1, len(daily_results))
    cagr = (nav.iloc[-1] / initial_nav) ** (252.0 / periods) - 1.0
    annual_vol = returns.std(ddof=0) * math.sqrt(252.0)
    sharpe = returns.mean() / returns.std(ddof=0) * math.sqrt(252.0) if returns.std(ddof=0) > EPS else 0.0
    downside = returns[returns < 0.0]
    sortino = returns.mean() / downside.std(ddof=0) * math.sqrt(252.0) if len(downside) and downside.std(ddof=0) > EPS else 0.0
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > EPS else 0.0
    return {
        "final_nav": float(nav.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "hit_ratio": float((returns > 0.0).mean()),
        "avg_turnover_value": float(daily_results["turnover_value"].mean()),
        "avg_turnover_ratio": float(daily_results["turnover_ratio"].mean()),
        "turnover": float(daily_results["turnover_ratio"].mean()),
        "total_turnover_value": float(daily_results["turnover_value"].sum()),
        "total_turnover_ratio": float(daily_results["turnover_ratio"].sum()),
        "total_commission": float(daily_results["commission"].sum()),
        "commission_drag": float(daily_results["commission"].sum() / initial_nav),
        "overnight_pnl_sum": float(daily_results["overnight_pnl"].sum()),
        "open_to_close_pnl_sum": float(daily_results["open_to_close_pnl"].sum()),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def build_trade_history_rows(
    strategy: str,
    decision_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    decision_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    old_quantities: Mapping[str, float],
    new_quantities: Mapping[str, float],
    target_sleeve_weights: Mapping[str, float],
    result: Mapping[str, float],
    stock_ratio: float,
) -> List[Mapping[str, object]]:
    if _a0_build_trade_rows is None:
        return []
    return _a0_build_trade_rows(
        strategy=strategy,
        decision_date=pd.Timestamp(decision_date).date().isoformat(),
        execution_date=pd.Timestamp(execution_date).date().isoformat(),
        decision_df=decision_df,
        execution_df=execution_df,
        old_qty=old_quantities,
        new_qty=new_quantities,
        target_sleeve_weights=target_sleeve_weights,
        result=result,
        stock_ratio=stock_ratio,
    )


def _save_common_outputs(
    out_dir: Path,
    daily_rows: Sequence[Mapping[str, object]],
    weight_rows: Sequence[Mapping[str, object]],
    trade_rows: Sequence[Mapping[str, object]],
    decision_rows: Sequence[Mapping[str, object]],
    expert_prob_rows: Sequence[Mapping[str, object]],
    expert_return_rows: Sequence[Mapping[str, object]],
    log_wealth_rows: Sequence[Mapping[str, object]],
    summary_extra: Mapping[str, object],
    strategy: str,
    top_k: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame(daily_rows)
    weights = pd.DataFrame(weight_rows)
    trades = pd.DataFrame(trade_rows)
    expert_probs = pd.DataFrame(expert_prob_rows)
    expert_returns = pd.DataFrame(expert_return_rows)
    log_wealth = pd.DataFrame(log_wealth_rows)

    daily.to_csv(out_dir / "daily_results.csv", index=False)
    weights.to_csv(out_dir / "weights.csv", index=False)
    trades.to_csv(out_dir / "trade_history.csv", index=False)
    expert_probs.to_csv(out_dir / "expert_probs.csv", index=False)
    expert_returns.to_csv(out_dir / "expert_gross_returns.csv", index=False)
    log_wealth.to_csv(out_dir / "a6_log_wealth.csv", index=False)
    relative_cols = ["decision_date", "execution_date", "a6_index_relative_log_wealth"]
    if log_wealth.empty:
        pd.DataFrame(columns=relative_cols).to_csv(out_dir / "a6_index_relative_log_wealth.csv", index=False)
    else:
        log_wealth[relative_cols].to_csv(out_dir / "a6_index_relative_log_wealth.csv", index=False)
    _write_jsonl(out_dir / "decision_log.jsonl", decision_rows)

    summary = {"strategy": strategy, "top_k": top_k}
    summary.update(compute_performance_metrics(daily) if not daily.empty else {})
    summary.update(summary_extra)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def run_A6_core_backtest(
    panel: pd.DataFrame,
    top_k: int,
    out_dir: str | Path,
    eta: float = 1.0,
    gamma: float = 0.99,
    config: Optional[BacktestConfig] = None,
) -> None:
    config = config or BacktestConfig()
    regime_path = default_regime_path()
    regime = load_regime_table(regime_path)
    kofr_path = default_kofr_path()
    kofr = load_kofr_table(kofr_path)
    kofr_path = default_kofr_path()
    kofr = load_kofr_table(kofr_path)
    kofr_path = default_kofr_path()
    kofr = load_kofr_table(kofr_path)
    panel = _panel_with_a5_labels(panel)
    dates = sorted(panel["date"].drop_duplicates())
    next_dates = _next_date_map(dates)
    dons_state = initialize_dons_state(CORE_EXPERTS, eta=eta, gamma=gamma)
    shadow_states = initialize_shadow_expert_states(CORE_EXPERTS, config.initial_nav)
    a6_state = initialize_portfolio_state(config.initial_nav)
    log_wealth_value = 0.0
    index_relative_log_wealth = 0.0

    daily_rows: List[Mapping[str, object]] = []
    weight_rows: List[Mapping[str, object]] = []
    trade_rows: List[Mapping[str, object]] = []
    decision_rows: List[Mapping[str, object]] = []
    expert_prob_rows: List[Mapping[str, object]] = []
    expert_return_rows: List[Mapping[str, object]] = []
    log_wealth_rows: List[Mapping[str, object]] = []

    for decision_date in dates[:-1]:
        execution_date = next_dates[decision_date]
        day_df = panel.loc[panel["date"] == decision_date].copy()
        next_day_df = panel.loc[panel["date"] == execution_date].copy()
        valid = get_valid_universe(day_df)
        if valid.empty:
            continue

        history = panel.loc[panel["date"] < decision_date].copy()
        expert_weights = make_core_expert_weights(day_df, history, top_k)
        expert_probs = compute_expert_mixture_probs(dons_state)
        blended_weights = blend_expert_weights(expert_weights, expert_probs, valid["ticker"].astype(str))
        execution_stock_ratio = regime_stock_ratio_for_date(regime, pd.Timestamp(decision_date), config.stock_ratio)
        execution_cash_return = kofr_cash_return_for_date(kofr, pd.Timestamp(execution_date))
        execution_config = replace(config, stock_ratio=execution_stock_ratio, cash_return=execution_cash_return)
        previous_nav = a6_state.nav
        old_quantities = dict(a6_state.quantities)
        a6_state, details = rebalance_and_mark_to_market_next_day(
            a6_state, blended_weights, day_df, next_day_df, execution_config
        )
        trade_rows.extend(
            build_trade_history_rows(
                strategy=f"A6_core_k{top_k}",
                decision_date=decision_date,
                execution_date=execution_date,
                decision_df=day_df,
                execution_df=next_day_df,
                old_quantities=old_quantities,
                new_quantities=a6_state.quantities,
                target_sleeve_weights=blended_weights,
                result=details,
                stock_ratio=execution_stock_ratio,
            )
        )
        index_gross = _index_gross_return(day_df, next_day_df)
        a6_gross = max(EPS, a6_state.nav / previous_nav)
        log_wealth_value += math.log(a6_gross)
        index_relative_log_wealth += math.log(a6_gross / max(EPS, index_gross))

        expert_gross_returns: Dict[str, float] = {}
        for expert_name in CORE_EXPERTS:
            next_shadow_state, gross_return = rebalance_and_mark_to_market_expert_shadow_state(
                shadow_states[expert_name], expert_weights[expert_name], day_df, next_day_df, execution_config
            )
            shadow_states[expert_name] = next_shadow_state
            expert_gross_returns[expert_name] = gross_return

        prob_row = {
            "decision_date": decision_date,
            "execution_date": execution_date,
            **expert_probs,
            "expert_probability_entropy": _entropy(expert_probs),
        }
        expert_prob_rows.append(prob_row)
        expert_return_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                **expert_gross_returns,
            }
        )

        daily_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "strategy": f"A6_core_k{top_k}",
                "nav_close": details["nav_close"],
                "nav_open_pre": details["nav_open_pre"],
                "nav_open_post": details["nav_open_post"],
                "overnight_pnl": details["overnight_pnl"],
                "open_to_close_pnl": details["open_to_close_pnl"],
                "commission": details["commission"],
                "turnover_value": details["turnover_value"],
                "turnover_ratio": details["turnover_ratio"],
                "daily_return": details["daily_return"],
                "stock_ratio": execution_stock_ratio,
                "cash_return": execution_cash_return,
                "fallback_used": False,
                "effective_num_positions": _effective_num_positions(blended_weights),
                "index_gross_return_close_to_close": index_gross,
            }
        )
        for _, row in valid.iterrows():
            ticker = str(row["ticker"])
            weight_rows.append(
                {
                    "decision_date": decision_date,
                    "execution_date": execution_date,
                    "ticker": ticker,
                    "ETF_id": row.get("ETF_id", ticker),
                    "sleeve_weight": blended_weights.get(ticker, 0.0),
                    "portfolio_weight": execution_stock_ratio * blended_weights.get(ticker, 0.0),
                }
            )
        decision_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "strategy": f"A6_core_k{top_k}",
                "expert_probs": expert_probs,
                "expert_net_gross_returns": expert_gross_returns,
                "dons_eta": eta,
                "dons_gamma": gamma,
            }
        )
        log_wealth_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "a6_log_wealth": log_wealth_value,
                "a6_index_relative_log_wealth": index_relative_log_wealth,
                "index_gross_return_close_to_close": index_gross,
            }
        )

        dons_state = update_dons_state(dons_state, expert_gross_returns)

    final_probs = compute_expert_mixture_probs(dons_state)
    prob_df = pd.DataFrame(expert_prob_rows)
    avg_probs = (
        {name: float(prob_df[name].mean()) for name in CORE_EXPERTS if name in prob_df.columns}
        if not prob_df.empty
        else final_probs
    )
    summary_extra = {
        "average_expert_probability": avg_probs,
        "final_expert_mixture_weight": final_probs,
        "best_expert_by_average_mixture_weight": max(avg_probs, key=avg_probs.get) if avg_probs else None,
        "expert_probability_entropy": _entropy(final_probs),
        "dons_eta": eta,
        "dons_gamma": gamma,
        "stock_ratio": config.stock_ratio,
        "stock_ratio_mode": "regime" if regime is not None else "constant",
        "regime_path": _display_path(regime_path),
        "cash_return_mode": "kofr_1_trading_day" if kofr is not None else "zero",
        "kofr_path": _display_path(kofr_path),
    }
    _save_common_outputs(
        Path(out_dir),
        daily_rows,
        weight_rows,
        trade_rows,
        decision_rows,
        expert_prob_rows,
        expert_return_rows,
        log_wealth_rows,
        summary_extra,
        f"A6_core_k{top_k}",
        top_k,
    )


def _load_external_weights(path: Path) -> pd.DataFrame:
    weights = pd.read_csv(path)
    weights["decision_date"] = pd.to_datetime(weights["decision_date"])
    weights["ticker"] = weights["ticker"].astype(str)
    return weights


def _load_external_returns(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path)
    daily["decision_date"] = pd.to_datetime(daily["decision_date"])
    if "execution_date" in daily.columns:
        daily["execution_date"] = pd.to_datetime(daily["execution_date"])
    if "net_gross_return" not in daily.columns:
        if "daily_return" in daily.columns:
            daily["net_gross_return"] = 1.0 + daily["daily_return"].astype(float)
        elif "nav_close" in daily.columns:
            daily["net_gross_return"] = daily["nav_close"].astype(float) / daily["nav_close"].astype(float).shift(1)
            daily["net_gross_return"] = daily["net_gross_return"].fillna(1.0 + daily.get("daily_return", 0.0))
        else:
            raise ValueError(f"Cannot infer expert net gross return from {path}")
    return daily


def load_external_expert_weights_and_returns(out_root: str | Path, top_k: int) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    root = Path(out_root)
    loaded: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for expert_name, dirname_template in FULL_EXPERT_DIRS.items():
        expert_dir = root / dirname_template.format(top_k=top_k)
        weights_path = expert_dir / "weights.csv"
        returns_path = expert_dir / "daily_results.csv"
        if not weights_path.exists() or not returns_path.exists():
            raise FileNotFoundError(f"Missing A6-full expert inputs for {expert_name}: {expert_dir}")
        loaded[expert_name] = (_load_external_weights(weights_path), _load_external_returns(returns_path))
    return loaded


def _external_weights_for_date(weights: pd.DataFrame, decision_date: pd.Timestamp, valid_tickers: Iterable[str]) -> Dict[str, float]:
    rows = weights.loc[weights["decision_date"] == decision_date]
    raw = dict(zip(rows["ticker"].astype(str), rows["sleeve_weight"].astype(float)))
    return normalize_sleeve_weights(raw, valid_tickers)


def _external_return_for_date(daily: pd.DataFrame, decision_date: pd.Timestamp) -> float:
    rows = daily.loc[daily["decision_date"] == decision_date]
    if rows.empty:
        return 1.0
    return max(EPS, float(rows.iloc[0]["net_gross_return"]))


def run_A6_full_backtest(
    panel: pd.DataFrame,
    top_k: int,
    out_dir: str | Path,
    expert_out_root: str | Path,
    eta: float = 1.0,
    gamma: float = 0.99,
    config: Optional[BacktestConfig] = None,
) -> None:
    config = config or BacktestConfig()
    regime_path = default_regime_path()
    regime = load_regime_table(regime_path)
    panel = _panel_with_a5_labels(panel)
    dates = sorted(panel["date"].drop_duplicates())
    next_dates = _next_date_map(dates)
    external = load_external_expert_weights_and_returns(expert_out_root, top_k)
    dons_state = initialize_dons_state(FULL_EXPERTS, eta=eta, gamma=gamma)
    rank_shadow_states = initialize_shadow_expert_states(
        ("MARKET_ONLY", "FLOW_ONLY", "ROTATION_ONLY"), config.initial_nav
    )
    a6_state = initialize_portfolio_state(config.initial_nav)
    log_wealth_value = 0.0
    index_relative_log_wealth = 0.0

    daily_rows: List[Mapping[str, object]] = []
    weight_rows: List[Mapping[str, object]] = []
    trade_rows: List[Mapping[str, object]] = []
    decision_rows: List[Mapping[str, object]] = []
    expert_prob_rows: List[Mapping[str, object]] = []
    expert_return_rows: List[Mapping[str, object]] = []
    log_wealth_rows: List[Mapping[str, object]] = []

    for decision_date in dates[:-1]:
        execution_date = next_dates[decision_date]
        day_df = panel.loc[panel["date"] == decision_date].copy()
        next_day_df = panel.loc[panel["date"] == execution_date].copy()
        valid = get_valid_universe(day_df)
        if valid.empty:
            continue
        valid_tickers = list(valid["ticker"].astype(str))
        history = panel.loc[panel["date"] < decision_date].copy()
        expert_weights = {
            "MARKET_ONLY": make_single_rank_expert(day_df, "market_score", top_k),
            "FLOW_ONLY": make_single_rank_expert(day_df, "flow_score", top_k),
            "ROTATION_ONLY": make_single_rank_expert(day_df, "rotation_score", top_k),
        }
        for expert_name, (weights_df, _) in external.items():
            expert_weights[expert_name] = _external_weights_for_date(weights_df, decision_date, valid_tickers)
        expert_weights.setdefault("A0_EQUAL_WEIGHT", make_equal_weight_expert(day_df))
        expert_weights.setdefault("A1_RULE_COMPOSITE", make_composite_expert(day_df, top_k))
        expert_weights.setdefault("A5_BAYESIAN_WINNER", make_A5_expert(day_df, history, top_k))

        expert_probs = compute_expert_mixture_probs(dons_state)
        blended_weights = blend_expert_weights(expert_weights, expert_probs, valid_tickers)
        execution_stock_ratio = regime_stock_ratio_for_date(regime, pd.Timestamp(decision_date), config.stock_ratio)
        execution_cash_return = kofr_cash_return_for_date(kofr, pd.Timestamp(execution_date))
        execution_config = replace(config, stock_ratio=execution_stock_ratio, cash_return=execution_cash_return)
        previous_nav = a6_state.nav
        old_quantities = dict(a6_state.quantities)
        a6_state, details = rebalance_and_mark_to_market_next_day(
            a6_state, blended_weights, day_df, next_day_df, execution_config
        )
        trade_rows.extend(
            build_trade_history_rows(
                strategy=f"A6_full_k{top_k}",
                decision_date=decision_date,
                execution_date=execution_date,
                decision_df=day_df,
                execution_df=next_day_df,
                old_quantities=old_quantities,
                new_quantities=a6_state.quantities,
                target_sleeve_weights=blended_weights,
                result=details,
                stock_ratio=execution_stock_ratio,
            )
        )
        index_gross = _index_gross_return(day_df, next_day_df)
        a6_gross = max(EPS, a6_state.nav / previous_nav)
        log_wealth_value += math.log(a6_gross)
        index_relative_log_wealth += math.log(a6_gross / max(EPS, index_gross))

        expert_gross_returns = {
            "MARKET_ONLY": 1.0,
            "FLOW_ONLY": 1.0,
            "ROTATION_ONLY": 1.0,
        }
        # Rank-only experts are not external outputs in the plan, so keep the same
        # persistent shadow accounting used by A6-core for these three experts.
        for expert_name in ("MARKET_ONLY", "FLOW_ONLY", "ROTATION_ONLY"):
            next_shadow_state, gross = rebalance_and_mark_to_market_expert_shadow_state(
                rank_shadow_states[expert_name], expert_weights[expert_name], day_df, next_day_df, execution_config
            )
            rank_shadow_states[expert_name] = next_shadow_state
            expert_gross_returns[expert_name] = gross
        for expert_name, (_, returns_df) in external.items():
            expert_gross_returns[expert_name] = _external_return_for_date(returns_df, decision_date)

        expert_prob_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                **expert_probs,
                "expert_probability_entropy": _entropy(expert_probs),
            }
        )
        expert_return_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                **expert_gross_returns,
            }
        )
        daily_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "strategy": f"A6_full_k{top_k}",
                "nav_close": details["nav_close"],
                "nav_open_pre": details["nav_open_pre"],
                "nav_open_post": details["nav_open_post"],
                "overnight_pnl": details["overnight_pnl"],
                "open_to_close_pnl": details["open_to_close_pnl"],
                "commission": details["commission"],
                "turnover_value": details["turnover_value"],
                "turnover_ratio": details["turnover_ratio"],
                "daily_return": details["daily_return"],
                "stock_ratio": execution_stock_ratio,
                "cash_return": execution_cash_return,
                "fallback_used": False,
                "effective_num_positions": _effective_num_positions(blended_weights),
                "index_gross_return_close_to_close": index_gross,
            }
        )
        for _, row in valid.iterrows():
            ticker = str(row["ticker"])
            weight_rows.append(
                {
                    "decision_date": decision_date,
                    "execution_date": execution_date,
                    "ticker": ticker,
                    "ETF_id": row.get("ETF_id", ticker),
                    "sleeve_weight": blended_weights.get(ticker, 0.0),
                    "portfolio_weight": execution_stock_ratio * blended_weights.get(ticker, 0.0),
                }
            )
        decision_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "strategy": f"A6_full_k{top_k}",
                "expert_probs": expert_probs,
                "expert_net_gross_returns": expert_gross_returns,
                "dons_eta": eta,
                "dons_gamma": gamma,
            }
        )
        log_wealth_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "a6_log_wealth": log_wealth_value,
                "a6_index_relative_log_wealth": index_relative_log_wealth,
                "index_gross_return_close_to_close": index_gross,
            }
        )
        dons_state = update_dons_state(dons_state, expert_gross_returns)

    final_probs = compute_expert_mixture_probs(dons_state)
    prob_df = pd.DataFrame(expert_prob_rows)
    avg_probs = (
        {name: float(prob_df[name].mean()) for name in FULL_EXPERTS if name in prob_df.columns}
        if not prob_df.empty
        else final_probs
    )
    summary_extra = {
        "average_expert_probability": avg_probs,
        "final_expert_mixture_weight": final_probs,
        "best_expert_by_average_mixture_weight": max(avg_probs, key=avg_probs.get) if avg_probs else None,
        "expert_probability_entropy": _entropy(final_probs),
        "dons_eta": eta,
        "dons_gamma": gamma,
        "stock_ratio": config.stock_ratio,
        "stock_ratio_mode": "regime" if regime is not None else "constant",
        "regime_path": _display_path(regime_path),
        "cash_return_mode": "kofr_1_trading_day" if kofr is not None else "zero",
        "kofr_path": _display_path(kofr_path),
    }
    _save_common_outputs(
        Path(out_dir),
        daily_rows,
        weight_rows,
        trade_rows,
        decision_rows,
        expert_prob_rows,
        expert_return_rows,
        log_wealth_rows,
        summary_extra,
        f"A6_full_k{top_k}",
        top_k,
    )


SELECTED_EXTERNAL_EXPERT_DIRS = {
    "A0": "A0",
    "A1": "A1_k{top_k}",
    "A2a": "A2a_k{top_k}",
    "A2b": "A2b_k{top_k}",
    "A3": "A3_k{top_k}",
    "A4": "A4_k{top_k}",
    "A5": "A5_k{top_k}",
}


def load_selected_external_experts(out_root: str | Path, top_k: int) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    root = Path(out_root)
    loaded: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for expert_name, dirname_template in SELECTED_EXTERNAL_EXPERT_DIRS.items():
        expert_dir = root / dirname_template.format(top_k=top_k)
        weights_path = expert_dir / "weights.csv"
        returns_path = expert_dir / "daily_results.csv"
        if not weights_path.exists() or not returns_path.exists():
            raise FileNotFoundError(f"Missing A6-selected expert inputs for {expert_name}: {expert_dir}")
        loaded[expert_name] = (_load_external_weights(weights_path), _load_external_returns(returns_path))
    return loaded


def run_A6_selected_external_backtest(
    panel: pd.DataFrame,
    top_k: int,
    out_dir: str | Path,
    expert_out_root: str | Path,
    eta: float = 1.0,
    gamma: float = 0.99,
    config: Optional[BacktestConfig] = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    config = config or BacktestConfig()
    regime_path = default_regime_path()
    regime = load_regime_table(regime_path)
    kofr_path = default_kofr_path()
    kofr = load_kofr_table(kofr_path)
    dates = sorted(panel["date"].drop_duplicates())
    next_dates = _next_date_map(dates)
    external = load_selected_external_experts(expert_out_root, top_k)
    expert_names = list(SELECTED_EXTERNAL_EXPERT_DIRS)
    dons_state = initialize_dons_state(expert_names, eta=eta, gamma=gamma)
    a6_state = initialize_portfolio_state(config.initial_nav)
    log_wealth_value = 0.0
    index_relative_log_wealth = 0.0

    daily_rows: List[Mapping[str, object]] = []
    weight_rows: List[Mapping[str, object]] = []
    trade_rows: List[Mapping[str, object]] = []
    decision_rows: List[Mapping[str, object]] = []
    expert_prob_rows: List[Mapping[str, object]] = []
    expert_return_rows: List[Mapping[str, object]] = []
    log_wealth_rows: List[Mapping[str, object]] = []
    strategy = "A6_DONS_A0_A1_A2A_A2B_A3_A4_A5"
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

    for decision_date in dates[:-1]:
        decision_ts = pd.Timestamp(decision_date)
        if start_ts is not None and decision_ts < start_ts:
            continue
        if end_ts is not None and decision_ts > end_ts:
            continue
        execution_date = next_dates[decision_date]
        day_df = panel.loc[panel["date"] == decision_date].copy()
        next_day_df = panel.loc[panel["date"] == execution_date].copy()
        valid = get_valid_universe(day_df)
        if valid.empty:
            continue
        valid_tickers = list(valid["ticker"].astype(str))
        expert_weights = {
            expert_name: _external_weights_for_date(weights_df, decision_date, valid_tickers)
            for expert_name, (weights_df, _) in external.items()
        }
        expert_probs = compute_expert_mixture_probs(dons_state)
        blended_weights = blend_expert_weights(expert_weights, expert_probs, valid_tickers)
        execution_stock_ratio = regime_stock_ratio_for_date(regime, pd.Timestamp(decision_date), config.stock_ratio)
        execution_cash_return = kofr_cash_return_for_date(kofr, pd.Timestamp(execution_date))
        execution_config = replace(config, stock_ratio=execution_stock_ratio, cash_return=execution_cash_return)
        previous_nav = a6_state.nav
        old_quantities = dict(a6_state.quantities)
        a6_state, details = rebalance_and_mark_to_market_next_day(
            a6_state, blended_weights, day_df, next_day_df, execution_config
        )
        trade_rows.extend(
            build_trade_history_rows(
                strategy=strategy,
                decision_date=decision_date,
                execution_date=execution_date,
                decision_df=day_df,
                execution_df=next_day_df,
                old_quantities=old_quantities,
                new_quantities=a6_state.quantities,
                target_sleeve_weights=blended_weights,
                result=details,
                stock_ratio=execution_stock_ratio,
            )
        )
        index_gross = _index_gross_return(day_df, next_day_df)
        a6_gross = max(EPS, a6_state.nav / previous_nav)
        log_wealth_value += math.log(a6_gross)
        index_relative_log_wealth += math.log(a6_gross / max(EPS, index_gross))
        expert_gross_returns = {
            expert_name: _external_return_for_date(returns_df, decision_date)
            for expert_name, (_, returns_df) in external.items()
        }
        expert_prob_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                **expert_probs,
                "expert_probability_entropy": _entropy(expert_probs),
            }
        )
        expert_return_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                **expert_gross_returns,
            }
        )
        daily_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "strategy": strategy,
                "nav_close": details["nav_close"],
                "nav_open_pre": details["nav_open_pre"],
                "nav_open_post": details["nav_open_post"],
                "overnight_pnl": details["overnight_pnl"],
                "open_to_close_pnl": details["open_to_close_pnl"],
                "commission": details["commission"],
                "turnover_value": details["turnover_value"],
                "turnover_ratio": details["turnover_ratio"],
                "daily_return": details["daily_return"],
                "stock_ratio": execution_stock_ratio,
                "cash_return": execution_cash_return,
                "fallback_used": False,
                "effective_num_positions": _effective_num_positions(blended_weights),
                "index_gross_return_close_to_close": index_gross,
            }
        )
        for _, row in valid.iterrows():
            ticker = str(row["ticker"])
            sleeve_weight = blended_weights.get(ticker, 0.0)
            weight_rows.append(
                {
                    "decision_date": decision_date,
                    "execution_date": execution_date,
                    "ticker": ticker,
                    "ETF_id": row.get("ETF_id", ticker),
                    "sleeve_weight": sleeve_weight,
                    "portfolio_weight": execution_stock_ratio * sleeve_weight,
                }
            )
        decision_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "strategy": strategy,
                "expert_probs": expert_probs,
                "expert_net_gross_returns": expert_gross_returns,
                "dons_eta": eta,
                "dons_gamma": gamma,
                "stock_ratio": execution_stock_ratio,
                "cash_return": execution_cash_return,
            }
        )
        log_wealth_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "a6_log_wealth": log_wealth_value,
                "a6_index_relative_log_wealth": index_relative_log_wealth,
                "index_gross_return_close_to_close": index_gross,
            }
        )
        dons_state = update_dons_state(dons_state, expert_gross_returns)

    final_probs = compute_expert_mixture_probs(dons_state)
    prob_df = pd.DataFrame(expert_prob_rows)
    avg_probs = (
        {name: float(prob_df[name].mean()) for name in expert_names if name in prob_df.columns}
        if not prob_df.empty
        else final_probs
    )
    summary_extra = {
        "experts": expert_names,
        "average_expert_probability": avg_probs,
        "final_expert_mixture_weight": final_probs,
        "best_expert_by_average_mixture_weight": max(avg_probs, key=avg_probs.get) if avg_probs else None,
        "expert_probability_entropy": _entropy(final_probs),
        "dons_eta": eta,
        "dons_gamma": gamma,
        "stock_ratio": config.stock_ratio,
        "stock_ratio_mode": "regime" if regime is not None else "constant",
        "regime_path": _display_path(regime_path),
        "cash_return_mode": "kofr_1_trading_day" if kofr is not None else "zero",
        "kofr_path": _display_path(kofr_path),
        "start_date": start_date,
        "end_date": end_date,
    }
    _save_common_outputs(
        Path(out_dir),
        daily_rows,
        weight_rows,
        trade_rows,
        decision_rows,
        expert_prob_rows,
        expert_return_rows,
        log_wealth_rows,
        summary_extra,
        strategy,
        top_k,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A6 D-ONS OBS expert allocator")
    parser.add_argument("--rank-panel", default="rank_panel.csv")
    parser.add_argument("--out-root", default="out")
    parser.add_argument("--mode", choices=("core", "full", "selected", "both"), default="core")
    parser.add_argument("--expert-out-root", default=None)
    parser.add_argument("--top-k", type=int, nargs="*", default=list(TOP_K_VALUES))
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial-nav", type=float, default=INITIAL_NAV)
    parser.add_argument("--commission-rate", type=float, default=COMMISSION_RATE)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_rank_panel(args.rank_panel)
    validate_backtest_panel(panel)
    config = BacktestConfig(initial_nav=args.initial_nav, commission_rate=args.commission_rate)
    out_root = Path(args.out_root)
    expert_out_root = Path(args.expert_out_root) if args.expert_out_root else out_root
    for top_k in args.top_k:
        if args.mode in ("core", "both"):
            run_A6_core_backtest(
                panel,
                top_k=top_k,
                out_dir=out_root / f"A6_core_k{top_k}",
                eta=args.eta,
                gamma=args.gamma,
                config=config,
            )
        if args.mode in ("full", "both"):
            run_A6_full_backtest(
                panel,
                top_k=top_k,
                out_dir=out_root / f"A6_full_k{top_k}",
                expert_out_root=expert_out_root,
                eta=args.eta,
                gamma=args.gamma,
                config=config,
            )
        if args.mode == "selected":
            run_A6_selected_external_backtest(
                panel,
                top_k=top_k,
                out_dir=out_root,
                expert_out_root=expert_out_root,
                eta=args.eta,
                gamma=args.gamma,
                config=config,
                start_date=args.start_date,
                end_date=args.end_date,
            )


if __name__ == "__main__":
    main()

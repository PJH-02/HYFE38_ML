from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


INITIAL_NAV = 1_000_000_000.0
COMMISSION_RATE = 0.00015
STOCK_RATIO = 1.0
TOP_K_GRID = (3, 5, 7, 10)
SCORE_COLUMNS = ("market_score", "flow_score", "rotation_score")


@dataclass(frozen=True)
class BacktestConfig:
    strategy: str
    initial_nav: float = INITIAL_NAV
    commission_rate: float = COMMISSION_RATE
    stock_ratio: float = STOCK_RATIO
    cash_return: float = 0.0
    fixed_point_tol: float = 1e-10
    fixed_point_max_iter: int = 50


def default_panel_path() -> Path:
    return Path(__file__).resolve().parent / "rank_panel.csv"


def default_out_root() -> Path:
    return Path(__file__).resolve().parent / "out"


def default_regime_path() -> Path | None:
    env_path = os.getenv("REGIME_CSV_PATH")
    if env_path:
        return Path(env_path)
    candidate = Path(__file__).resolve().parent / "regime_weights.csv"
    return candidate if candidate.exists() else None


def default_kofr_path() -> Path | None:
    env_path = os.getenv("KOFR_XLSX_PATH")
    if env_path:
        return Path(env_path)
    candidate = Path(__file__).resolve().parent.parent / "KOFR_20260508.xlsx"
    return candidate if candidate.exists() else None


def load_kofr_table(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    kofr = pd.read_excel(
        path,
        skiprows=4,
        names=["date", "kofr", "avg30", "avg90", "avg180", "kofr_index"],
    )
    kofr["date"] = pd.to_datetime(kofr["date"], format="%Y.%m.%d", errors="coerce")
    kofr["kofr"] = pd.to_numeric(kofr["kofr"], errors="coerce")
    kofr = kofr.dropna(subset=["date", "kofr"]).sort_values("date")
    if kofr.empty:
        raise ValueError("KOFR file has no valid daily KOFR rows")
    return kofr[["date", "kofr"]]


def kofr_cash_return_for_date(kofr: pd.DataFrame | None, date: pd.Timestamp, default: float = 0.0) -> float:
    if kofr is None:
        return float(default)
    eligible = kofr.loc[kofr["date"] <= pd.Timestamp(date)]
    if eligible.empty:
        return float(default)
    return float(eligible.iloc[-1]["kofr"]) / 100.0 / 252.0


def load_regime_table(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    regime = pd.read_csv(path)
    required = {"date", "cash_weight", "stock_weight"}
    missing = required - set(regime.columns)
    if missing:
        raise ValueError(f"Regime CSV missing columns: {sorted(missing)}")
    regime["date"] = pd.to_datetime(regime["date"])
    regime["stock_weight"] = pd.to_numeric(regime["stock_weight"], errors="coerce")
    regime = regime.dropna(subset=["date", "stock_weight"]).sort_values("date")
    if regime.empty:
        raise ValueError("Regime CSV has no valid rows")
    return regime


def regime_stock_ratio_for_date(regime: pd.DataFrame | None, date: pd.Timestamp, default: float) -> float:
    if regime is None:
        return float(default)
    eligible = regime.loc[regime["date"] <= pd.Timestamp(date)]
    if eligible.empty:
        return float(default)
    return float(eligible.iloc[-1]["stock_weight"])


def regime_info_for_date(regime: pd.DataFrame | None, date: pd.Timestamp) -> dict[str, Any] | None:
    if regime is None:
        return None
    eligible = regime.loc[regime["date"] <= pd.Timestamp(date)]
    if eligible.empty:
        return None
    row = eligible.iloc[-1]
    return {
        "regime_date": pd.Timestamp(row["date"]).date().isoformat(),
        "regime_key": str(row.get("regime_key", "")),
        "regime_name": str(row.get("regime_name", "")),
        "cash_weight": float(row.get("cash_weight", 1.0 - row["stock_weight"])),
        "stock_weight": float(row["stock_weight"]),
    }


def is_risk_on_regime(regime_info: Mapping[str, Any] | None) -> bool:
    if not regime_info:
        return False
    text = f"{regime_info.get('regime_key', '')} {regime_info.get('regime_name', '')}".lower()
    normalized = text.replace("-", "_").replace(" ", "_")
    return "risk_on" in normalized and "risk_off" not in normalized


def apply_regime_market_score(day_df: pd.DataFrame, regime_info: Mapping[str, Any] | None = None) -> pd.DataFrame:
    adjusted = day_df.copy()
    if "market_score_regime_adjusted" in adjusted.columns and adjusted["market_score_regime_adjusted"].fillna(False).astype(bool).all():
        return adjusted
    if "market_score_original" not in adjusted.columns:
        adjusted["market_score_original"] = adjusted["market_score"]
    if "market_rank_original" not in adjusted.columns and "market_rank" in adjusted.columns:
        adjusted["market_rank_original"] = adjusted["market_rank"]

    risk_on = is_risk_on_regime(regime_info)
    score = pd.to_numeric(adjusted["market_score_original"], errors="coerce")
    adjusted["market_score"] = (1.0 - score) if risk_on else score
    valid_mask = adjusted["market_score"].notna()
    adjusted.loc[valid_mask, "market_rank"] = adjusted.loc[valid_mask, "market_score"].rank(
        method="first", ascending=False
    )
    adjusted["market_score_regime_adjusted"] = True
    adjusted["market_score_regime_rule"] = "risk_on_invert_low_corr_score" if risk_on else "risk_off_neutral_keep_low_corr_score"
    if regime_info:
        adjusted["market_score_regime_key"] = str(regime_info.get("regime_key", ""))
    return adjusted


def apply_regime_market_score_by_next_execution(panel: pd.DataFrame, regime: pd.DataFrame | None) -> pd.DataFrame:
    dates = list(pd.Series(panel["date"].drop_duplicates()).sort_values())
    pieces = []
    for idx, decision_date in enumerate(dates):
        execution_date = dates[idx + 1] if idx + 1 < len(dates) else decision_date
        regime_info = regime_info_for_date(regime, pd.Timestamp(execution_date))
        pieces.append(apply_regime_market_score(panel.loc[panel["date"] == decision_date], regime_info))
    return pd.concat(pieces, ignore_index=True) if pieces else panel.copy()


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rank_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    if "ETF_id" not in panel.columns:
        panel["ETF_id"] = panel["ticker"]
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def validate_backtest_panel(panel: pd.DataFrame, require_index_close: bool = False) -> None:
    required = {
        "date",
        "ticker",
        "ETF_id",
        "open",
        "close",
        "tradable",
        "all_rank_valid",
        *SCORE_COLUMNS,
    }
    if require_index_close:
        required.add("index_close")
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"rank_panel.csv is missing required columns: {missing}")
    if panel.duplicated(["date", "ticker"]).any():
        raise ValueError("rank_panel.csv contains duplicate date,ticker rows")
    for col in ("open", "close"):
        if panel[col].isna().any() or (panel[col] <= 0).any():
            raise ValueError(f"{col} must be positive and non-null")
    forbidden = [
        col
        for col in panel.columns
        if any(token in col.lower() for token in ("future", "winner_label", "next_open", "next_close"))
    ]
    if forbidden:
        raise ValueError(f"rank_panel.csv must not contain future/label columns: {forbidden}")


def get_valid_universe(day_df: pd.DataFrame) -> pd.DataFrame:
    mask = day_df["tradable"].astype(bool) & day_df["all_rank_valid"].astype(bool)
    for col in SCORE_COLUMNS:
        if col in day_df.columns:
            mask &= day_df[col].notna()
    return day_df.loc[mask].copy()


def normalize_sleeve_weights(raw_weights: pd.Series, valid_tickers: pd.Index | list[str]) -> tuple[pd.Series, bool]:
    tickers = pd.Index([str(t) for t in valid_tickers])
    if len(tickers) == 0:
        return pd.Series(dtype=float), True
    weights = raw_weights.copy()
    weights.index = weights.index.astype(str)
    weights = weights.reindex(tickers).fillna(0.0).astype(float)
    weights = weights.clip(lower=0.0)
    total = float(weights.sum())
    fallback_used = not np.isfinite(total) or total <= 0.0
    if fallback_used:
        weights = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    else:
        weights = weights / total
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("normalized sleeve weights do not sum to 1")
    return weights, fallback_used


def apply_top_k_filter(weights: pd.Series, top_k: int) -> pd.Series:
    weights = weights.astype(float).clip(lower=0.0)
    if top_k < 10 and len(weights) > top_k:
        keep = weights.sort_values(ascending=False).head(top_k).index
        weights = weights.where(weights.index.isin(keep), 0.0)
    total = weights.sum()
    return weights / total if total > 0 else weights


def score_proportional_weights(
    day_df: pd.DataFrame,
    score_col: str,
    top_k: int,
) -> tuple[pd.Series, bool, pd.DataFrame]:
    valid = get_valid_universe(day_df)
    if valid.empty:
        return pd.Series(dtype=float), True, valid
    scored = valid.assign(_score=valid[score_col].fillna(0.0).astype(float).clip(lower=0.0))
    if top_k < 10:
        scored = scored.sort_values(["_score", "ticker"], ascending=[False, True]).head(top_k)
    raw = pd.Series(scored["_score"].to_numpy(), index=scored["ticker"].astype(str))
    weights, fallback = normalize_sleeve_weights(raw, scored["ticker"].astype(str))
    return weights, fallback, scored.drop(columns=["_score"])


def generate_equal_weight(day_df: pd.DataFrame, top_k: int | None = None) -> tuple[pd.Series, bool]:
    valid = get_valid_universe(day_df)
    if valid.empty:
        return pd.Series(dtype=float), True
    raw = pd.Series(1.0, index=valid["ticker"].astype(str))
    return normalize_sleeve_weights(raw, valid["ticker"].astype(str))


def _next_day_prices(panel: pd.DataFrame, dates: list[pd.Timestamp], idx: int) -> pd.DataFrame:
    return panel.loc[panel["date"] == dates[idx + 1]].set_index("ticker", drop=False)


def floor_target_values(target_budget: pd.Series, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    aligned_prices = prices.reindex(target_budget.index).astype(float)
    raw_qty = target_budget.astype(float) / aligned_prices
    qty = raw_qty.map(lambda value: math.floor(value) if np.isfinite(value) and value > 0 else 0)
    qty = qty.astype(float)
    values = qty * aligned_prices
    return qty, values


def _float_series(values: pd.Series | Mapping[str, float] | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.Series):
        out = values.copy()
    else:
        out = pd.Series(dict(values), dtype=float)
    out.index = out.index.astype(str)
    return out.astype(float)


def build_trade_rows(
    strategy: str,
    decision_date: str,
    execution_date: str,
    decision_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    old_qty: pd.Series | Mapping[str, float],
    new_qty: pd.Series | Mapping[str, float],
    target_sleeve_weights: pd.Series | Mapping[str, float],
    result: Mapping[str, float],
    stock_ratio: float,
) -> list[dict[str, Any]]:
    old_qty_s = _float_series(old_qty)
    new_qty_s = _float_series(new_qty)
    target_s = _float_series(target_sleeve_weights)
    prices = execution_df.copy()
    if "ticker" in prices.columns:
        prices["ticker"] = prices["ticker"].astype(str)
        prices = prices.set_index("ticker", drop=False)
    else:
        prices.index = prices.index.astype(str)
    meta = decision_df.copy()
    meta["ticker"] = meta["ticker"].astype(str)
    meta = meta.drop_duplicates("ticker").set_index("ticker", drop=False)

    tickers = sorted(set(old_qty_s.index) | set(new_qty_s.index) | set(target_s.index))
    nav_open_pre = float(result.get("nav_open_pre", 0.0))
    nav_open_post = float(result.get("nav_open_post", 0.0))
    nav_close = float(result.get("nav_close", 0.0))
    day_commission = float(result.get("commission", 0.0))
    day_turnover = float(result.get("turnover_value", 0.0))
    daily_return = float(result.get("daily_return", 0.0))
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        if ticker not in prices.index:
            continue
        open_price = float(prices.at[ticker, "open"])
        close_price = float(prices.at[ticker, "close"])
        prev_qty = float(old_qty_s.get(ticker, 0.0))
        target_qty = float(new_qty_s.get(ticker, 0.0))
        prev_value_open = prev_qty * open_price
        target_value_open = target_qty * open_price
        delta_value_open = target_value_open - prev_value_open
        trade_turnover = abs(delta_value_open)
        sleeve_weight = float(target_s.get(ticker, 0.0))
        target_portfolio_weight = stock_ratio * sleeve_weight
        commission_allocated = day_commission * trade_turnover / day_turnover if day_turnover > 0 else 0.0
        etf_id = str(meta.at[ticker, "ETF_id"]) if ticker in meta.index and "ETF_id" in meta.columns else ticker
        name = str(meta.at[ticker, "name"]) if ticker in meta.index and "name" in meta.columns else ticker
        delta_qty = target_qty - prev_qty
        rows.append(
            {
                "strategy": strategy,
                "decision_date": decision_date,
                "execution_date": execution_date,
                "ticker": ticker,
                "ETF_id": etf_id,
                "name": name,
                "open": open_price,
                "close": close_price,
                "prev_qty": prev_qty,
                "target_qty": target_qty,
                "delta_qty": delta_qty,
                "side": "BUY" if delta_qty > 0 else ("SELL" if delta_qty < 0 else "HOLD"),
                "prev_value_open": prev_value_open,
                "target_value_open": target_value_open,
                "delta_value_open": delta_value_open,
                "trade_turnover_value": trade_turnover,
                "day_turnover_value": day_turnover,
                "commission_allocated": commission_allocated,
                "day_commission": day_commission,
                "nav_open_pre": nav_open_pre,
                "nav_open_post": nav_open_post,
                "nav_close": nav_close,
                "daily_return": daily_return,
                "stock_ratio": stock_ratio,
                "cash_weight": 1.0 - stock_ratio,
                "target_sleeve_weight": sleeve_weight,
                "target_portfolio_weight": target_portfolio_weight,
                "prev_portfolio_weight_open": prev_value_open / nav_open_pre if nav_open_pre > 0 else 0.0,
                "actual_portfolio_weight_open": target_value_open / nav_open_post if nav_open_post > 0 else 0.0,
                "cash_after_rebalance": nav_open_post - float((new_qty_s.reindex(prices.index).fillna(0.0) * prices["open"].astype(float)).sum()),
                "rebalanced": day_turnover > 0,
            }
        )
    return rows


def rebalance_and_mark_to_market_next_day(
    state: dict,
    target_sleeve_weights: pd.Series,
    price_t1: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[dict, dict]:
    old_qty = state["qty"].copy()
    cash = float(state["cash"])
    prev_nav_close = float(state["nav_close"])

    missing_held = old_qty.index.difference(price_t1.index)
    if len(missing_held) > 0:
        raise ValueError(f"next-day prices are missing for held tickers: {list(missing_held)}")
    missing_target = target_sleeve_weights.index.difference(price_t1.index)
    if len(missing_target) > 0:
        raise ValueError(f"next-day prices are missing for target tickers: {list(missing_target)}")

    executable = target_sleeve_weights.index
    target_sleeve_weights = target_sleeve_weights.reindex(executable).fillna(0.0)
    if len(target_sleeve_weights) > 0 and target_sleeve_weights.sum() > 0:
        target_sleeve_weights = target_sleeve_weights / target_sleeve_weights.sum()

    all_tickers = old_qty.index.union(target_sleeve_weights.index).intersection(price_t1.index)
    old_qty = old_qty.reindex(all_tickers).fillna(0.0)
    open_prices = price_t1.loc[all_tickers, "open"].astype(float)
    close_prices = price_t1.loc[all_tickers, "close"].astype(float)
    old_value_open = old_qty * open_prices

    nav_open_pre = cash + float(old_value_open.sum())
    overnight_pnl = nav_open_pre - prev_nav_close

    target_weights = target_sleeve_weights.reindex(all_tickers).fillna(0.0)
    nav_open_post = nav_open_pre
    commission = 0.0
    target_values = pd.Series(0.0, index=all_tickers)
    target_qty = pd.Series(0.0, index=all_tickers)
    for _ in range(config.fixed_point_max_iter):
        target_budget = config.stock_ratio * nav_open_post * target_weights
        target_qty, target_values = floor_target_values(target_budget, open_prices)
        turnover_value = float((target_values - old_value_open).abs().sum())
        commission_next = config.commission_rate * turnover_value
        nav_next = nav_open_pre - commission_next
        if abs(nav_next - nav_open_post) <= config.fixed_point_tol * max(1.0, nav_open_pre):
            nav_open_post = nav_next
            commission = commission_next
            break
        nav_open_post = nav_next
        commission = commission_next
    else:
        target_budget = config.stock_ratio * nav_open_post * target_weights
        target_qty, target_values = floor_target_values(target_budget, open_prices)
        turnover_value = float((target_values - old_value_open).abs().sum())
        commission = config.commission_rate * turnover_value
        nav_open_post = nav_open_pre - commission

    turnover_value = float((target_values - old_value_open).abs().sum())
    new_qty = target_qty
    cash_open_post = nav_open_post - float(target_values.sum())
    cash_close = cash_open_post * (1.0 + config.cash_return)
    nav_close = cash_close + float((new_qty * close_prices).sum())
    open_to_close_pnl = nav_close - nav_open_post

    new_state = {
        "cash": cash_close,
        "qty": new_qty[new_qty.abs() > 1e-12],
        "nav_close": nav_close,
    }
    result = {
        "nav_close": nav_close,
        "nav_open_pre": nav_open_pre,
        "nav_open_post": nav_open_post,
        "overnight_pnl": overnight_pnl,
        "open_to_close_pnl": open_to_close_pnl,
        "commission": commission,
        "turnover_value": turnover_value,
        "turnover_ratio": turnover_value / nav_open_pre if nav_open_pre else 0.0,
        "daily_return": nav_close / prev_nav_close - 1.0 if prev_nav_close else 0.0,
        "stock_ratio": config.stock_ratio,
        "cash_return": config.cash_return,
        "effective_num_positions": int((target_weights > 0).sum()),
    }
    return new_state, result


WeightGenerator = Callable[[pd.DataFrame], tuple[pd.Series, bool]]


def run_backtest(
    panel: pd.DataFrame,
    weight_generator: WeightGenerator,
    config: BacktestConfig,
    out_dir: Path,
    panel_path: Path | None = None,
    extra_summary: dict | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = list(panel["date"].drop_duplicates().sort_values())
    regime_path = default_regime_path()
    regime = load_regime_table(regime_path)
    kofr_path = default_kofr_path()
    kofr = load_kofr_table(kofr_path)
    state = {"cash": config.initial_nav, "qty": pd.Series(dtype=float), "nav_close": config.initial_nav}
    daily_rows: list[dict] = []
    weight_rows: list[dict] = []
    trade_rows: list[dict] = []
    log_rows: list[dict] = []
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

    for idx, decision_date in enumerate(dates[:-1]):
        decision_ts = pd.Timestamp(decision_date)
        if start_ts is not None and decision_ts < start_ts:
            continue
        if end_ts is not None and decision_ts > end_ts:
            continue
        execution_date = dates[idx + 1]
        regime_info = regime_info_for_date(regime, execution_date)
        day_df = apply_regime_market_score(panel.loc[panel["date"] == decision_date], regime_info)
        if get_valid_universe(day_df).empty:
            continue
        price_t1 = _next_day_prices(panel, dates, idx)
        target_weights, fallback_used = weight_generator(day_df)
        target_weights = target_weights[target_weights > 0]
        execution_stock_ratio = regime_stock_ratio_for_date(regime, execution_date, config.stock_ratio)
        execution_cash_return = kofr_cash_return_for_date(kofr, execution_date)
        execution_config = replace(config, stock_ratio=execution_stock_ratio, cash_return=execution_cash_return)
        old_qty = state["qty"].copy()
        state, result = rebalance_and_mark_to_market_next_day(state, target_weights, price_t1, execution_config)
        trade_rows.extend(
            build_trade_rows(
                strategy=config.strategy,
                decision_date=decision_date.date().isoformat(),
                execution_date=execution_date.date().isoformat(),
                decision_df=day_df,
                execution_df=price_t1,
                old_qty=old_qty,
                new_qty=state["qty"],
                target_sleeve_weights=target_weights,
                result=result,
                stock_ratio=execution_stock_ratio,
            )
        )

        row = {
            "decision_date": decision_date.date().isoformat(),
            "execution_date": execution_date.date().isoformat(),
            "strategy": config.strategy,
            **result,
            "fallback_used": bool(fallback_used),
        }
        daily_rows.append(row)

        etf_ids = day_df.set_index(day_df["ticker"].astype(str))["ETF_id"].astype(str).to_dict()
        for ticker, weight in target_weights.items():
            weight_rows.append(
                {
                    "decision_date": decision_date.date().isoformat(),
                    "execution_date": execution_date.date().isoformat(),
                    "ticker": str(ticker),
                    "ETF_id": etf_ids.get(str(ticker), str(ticker)),
                    "sleeve_weight": float(weight),
                    "portfolio_weight": float(execution_stock_ratio * weight),
                }
            )
        log_rows.append(
            {
                "decision_date": decision_date.date().isoformat(),
                "execution_date": execution_date.date().isoformat(),
                "strategy": config.strategy,
                "fallback_used": bool(fallback_used),
                "num_target_positions": int((target_weights > 0).sum()),
                "nav_close": float(result["nav_close"]),
            }
        )

    daily = pd.DataFrame(daily_rows)
    weights = pd.DataFrame(weight_rows)
    trades = pd.DataFrame(trade_rows)
    summary = compute_performance_metrics(daily, config.initial_nav)
    summary.update(
        {
            "strategy": config.strategy,
            "initial_nav": config.initial_nav,
            "commission_rate": config.commission_rate,
            "stock_ratio": config.stock_ratio,
            "stock_ratio_mode": "regime" if regime is not None else "constant",
            "regime_path": str(regime_path) if regime_path else None,
            "cash_return_mode": "kofr_1_trading_day" if kofr is not None else "zero",
            "kofr_path": str(kofr_path) if kofr_path else None,
            "rank_panel_hash": file_sha256(panel_path) if panel_path else None,
            "start_date": start_date,
            "end_date": end_date,
        }
    )
    if extra_summary:
        summary.update(extra_summary)

    daily.to_csv(out_dir / "daily_results.csv", index=False)
    weights.to_csv(out_dir / "weights.csv", index=False)
    trades.to_csv(out_dir / "trade_history.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_ready(summary), f, indent=2, sort_keys=True)
    with (out_dir / "decision_log.jsonl").open("w", encoding="utf-8") as f:
        for row in log_rows:
            f.write(json.dumps(_json_ready(row), sort_keys=True) + "\n")
    return daily, weights, summary


def compute_performance_metrics(daily_results: pd.DataFrame, initial_nav: float = INITIAL_NAV) -> dict:
    if daily_results.empty:
        return {
            "final_nav": initial_nav,
            "total_return": 0.0,
            "cagr": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "hit_ratio": 0.0,
            "avg_turnover_value": 0.0,
            "avg_turnover_ratio": 0.0,
            "turnover": 0.0,
            "total_turnover_value": 0.0,
            "total_turnover_ratio": 0.0,
            "total_commission": 0.0,
            "commission_drag": 0.0,
            "fallback_rate": 0.0,
            "overnight_pnl_sum": 0.0,
            "open_to_close_pnl_sum": 0.0,
        }
    returns = daily_results["daily_return"].astype(float)
    nav = daily_results["nav_close"].astype(float)
    final_nav = float(nav.iloc[-1])
    total_return = final_nav / initial_nav - 1.0
    years = max(len(daily_results) / 252.0, 1.0 / 252.0)
    cagr = (final_nav / initial_nav) ** (1.0 / years) - 1.0
    annual_vol = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe = float((returns.mean() * 252) / annual_vol) if annual_vol > 0 else 0.0
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(252)) if len(downside) else 0.0
    sortino = float((returns.mean() * 252) / downside_vol) if downside_vol > 0 else 0.0
    drawdown = nav / nav.cummax() - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    total_turnover = float(daily_results["turnover_value"].sum())
    total_commission = float(daily_results["commission"].sum())
    return {
        "final_nav": final_nav,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_ratio": float((returns > 0).mean()),
        "avg_turnover_value": float(daily_results["turnover_value"].mean()),
        "avg_turnover_ratio": float(daily_results["turnover_ratio"].mean()),
        "turnover": float(daily_results["turnover_ratio"].mean()),
        "total_turnover_value": total_turnover,
        "total_turnover_ratio": float(total_turnover / initial_nav),
        "total_commission": total_commission,
        "commission_drag": float(total_commission / initial_nav),
        "fallback_rate": float(daily_results["fallback_used"].astype(bool).mean()),
        "overnight_pnl_sum": float(daily_results["overnight_pnl"].sum()),
        "open_to_close_pnl_sum": float(daily_results["open_to_close_pnl"].sum()),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def run_A0_backtest(
    panel_path: Path | None = None,
    out_root: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel_path = panel_path or default_panel_path()
    out_root = out_root or default_out_root()
    panel = load_rank_panel(panel_path)
    validate_backtest_panel(panel)
    return run_backtest(
        panel=panel,
        weight_generator=lambda day: generate_equal_weight(day),
        config=BacktestConfig(strategy="A0_EQUAL_WEIGHT"),
        out_dir=out_root / "A0",
        panel_path=panel_path,
        start_date=start_date,
        end_date=end_date,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A0 equal-weight ETF baseline backtest.")
    parser.add_argument("--panel", type=Path, default=default_panel_path())
    parser.add_argument("--out-root", type=Path, default=default_out_root())
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()
    run_A0_backtest(args.panel, args.out_root, args.start_date, args.end_date)


if __name__ == "__main__":
    main()

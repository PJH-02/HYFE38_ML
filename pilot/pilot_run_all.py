from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TESTING_DIR = PROJECT_DIR / "testing"
WORK_DIR = BASE_DIR / "work"
OUT_DIR = BASE_DIR / "out"
DEFAULT_REGIME_FILENAME = "ml8_risk_regime_weights_2024_2025_weekly.csv"

if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from A0_equal_weight import (  # noqa: E402
    COMMISSION_RATE,
    apply_regime_market_score,
    apply_regime_market_score_by_next_execution,
    build_trade_rows,
    generate_equal_weight,
    get_valid_universe,
)
from A1_rule_based_rank_allocator import generate_rule_based_weight  # noqa: E402
from A2a_llm_opaque_rank_allocator import create_id_mapping, generate_A2a_weight  # noqa: E402
from A2b_llm_semantic_rank_allocator import generate_A2b_weight  # noqa: E402
from A3_llm_policy_pack_allocator import generate_A3_weight  # noqa: E402
from A4_rule_based_llm_blend import generate_A4_weight  # noqa: E402
from A5_bayesian_winner_loser_allocator import (  # noqa: E402
    bucketize_scores,
    generate_A5_weight,
    get_bayesian_history,
    make_state_key,
    prepare_A5_panel,
)


PILOT_STRATEGIES = ("A0", "A1", "A2", "A3", "A4", "A5")
INITIAL_NAV = 100_000_000.0
STOCK_RATIO = 1.0


def load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.split("#", 1)[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_testing_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TESTING_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def read_csv_or_fallback(path: Path, fallback: Path) -> Path:
    return path if path.exists() else fallback


def next_calendar_date(date_text: str) -> str:
    return (pd.Timestamp(date_text) + pd.Timedelta(days=1)).date().isoformat()


def latest_csv_date(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        dates = pd.read_csv(path, usecols=["date"], parse_dates=["date"])["date"].dropna()
    except Exception:
        return None
    return dates.max().date().isoformat() if not dates.empty else None


def csv_contains_date(path: Path, date_text: str) -> bool:
    if not path.exists():
        return False
    try:
        dates = pd.read_csv(path, usecols=["date"], dtype=str)["date"].astype(str)
    except Exception:
        return False
    return date_text in set(dates)


def ensure_recent_data(args: argparse.Namespace) -> None:
    if args.skip_fetch:
        return
    price_has = csv_contains_date(args.price_path, args.as_of_date)
    flow_has = csv_contains_date(args.flow_path, args.as_of_date)
    if price_has and flow_has:
        return
    base_price = args.price_path if args.price_path.exists() else TESTING_DIR / "sector_all_merged.csv"
    latest = latest_csv_date(base_price) or latest_csv_date(TESTING_DIR / "sector_all_merged.csv")
    from_date = args.fetch_from_date or (next_calendar_date(latest) if latest else args.start_date)
    command = [
        sys.executable,
        str(BASE_DIR / "kiwoom_fetch_data.py"),
        "--from-date",
        from_date,
        "--to-date",
        args.as_of_date,
        "--price-out",
        str(args.price_path),
        "--flow-out",
        str(args.flow_path),
    ]
    print(f"Fetching missing pilot data from {from_date} to {args.as_of_date}...", flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def find_default_regime_path() -> Path | None:
    local = BASE_DIR / "data" / "regime_weights.csv"
    if local.exists():
        return local
    try:
        matches = list((Path.home() / "Documents").rglob(DEFAULT_REGIME_FILENAME))
    except Exception:
        matches = []
    return matches[0] if matches else None


def load_regime_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("1900-01-01"),
                    "regime_key": "DEFAULT_FULLY_INVESTED",
                    "regime_name": "Default Fully Invested",
                    "cash_weight": 0.0,
                    "stock_weight": 1.0,
                }
            ]
        )
    regime = pd.read_csv(path)
    required = {"date", "cash_weight", "stock_weight"}
    missing = required - set(regime.columns)
    if missing:
        raise ValueError(f"Regime CSV missing columns: {sorted(missing)}")
    regime["date"] = pd.to_datetime(regime["date"])
    regime["cash_weight"] = pd.to_numeric(regime["cash_weight"], errors="coerce")
    regime["stock_weight"] = pd.to_numeric(regime["stock_weight"], errors="coerce")
    regime = regime.dropna(subset=["date", "cash_weight", "stock_weight"]).sort_values("date")
    if regime.empty:
        raise ValueError("Regime CSV has no valid rows")
    return regime


def regime_for_date(regime: pd.DataFrame, date_text: str) -> dict[str, Any]:
    target = pd.Timestamp(date_text)
    eligible = regime.loc[regime["date"] <= target]
    row = eligible.iloc[-1] if not eligible.empty else regime.iloc[0]
    return {
        "regime_date": pd.Timestamp(row["date"]).date().isoformat(),
        "regime_key": str(row.get("regime_key", "")),
        "regime_name": str(row.get("regime_name", "")),
        "cash_weight": float(row["cash_weight"]),
        "stock_weight": float(row["stock_weight"]),
    }


def build_rank_panel(price_path: Path, flow_path: Path, work_dir: Path = WORK_DIR) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    market_mod = load_testing_module("pilot_market_rank", "01_make_market_corr_rank.py")
    flow_mod = load_testing_module("pilot_flow_rank", "02_make_flow_price_rank.py")
    rotation_mod = load_testing_module("pilot_rotation_rank", "03_make_rotation_rank.py")
    merge_mod = load_testing_module("pilot_merge_rank_panel", "04_merge_rank_panel.py")
    market_path = work_dir / "market_rank.csv"
    flow_rank_path = work_dir / "flow_rank.csv"
    rotation_path = work_dir / "rotation_rank.csv"
    panel_path = work_dir / "rank_panel.csv"
    mapping_path = work_dir / "id_mapping.csv"
    market_mod.build_market_rank(price_path, market_path)
    flow_mod.build_flow_rank(price_path, flow_path, flow_rank_path)
    rotation_mod.build_rotation_rank(price_path, rotation_path)
    merge_mod.build_rank_panel(price_path, market_path, flow_rank_path, rotation_path, panel_path, mapping_path)
    return panel_path


def load_panel(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_path, dtype={"ticker": "string"})
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def parse_strategy_list(raw: str) -> list[str]:
    values = [part.strip().upper() for part in raw.split(",") if part.strip()]
    bad = sorted(set(values) - set(PILOT_STRATEGIES))
    if bad:
        raise ValueError(f"Unknown pilot strategies: {bad}")
    return values


def normalize_weights(weights: Any) -> dict[str, float]:
    if isinstance(weights, tuple):
        weights = weights[0]
    if hasattr(weights, "items"):
        raw = {str(k): max(float(v), 0.0) for k, v in weights.items()}
    else:
        raw = {}
    total = sum(raw.values())
    return {ticker: value / total for ticker, value in raw.items() if value > 0.0} if total > 0 else {}


def current_close_weights(state: dict[str, Any], day_df: pd.DataFrame) -> dict[str, float]:
    qty = {str(k): float(v) for k, v in state.get("qty", {}).items()}
    close = dict(zip(day_df["ticker"].astype(str), day_df["close"].astype(float)))
    values = {ticker: amount * close.get(ticker, 0.0) for ticker, amount in qty.items()}
    stock_value = sum(values.values())
    return {ticker: value / stock_value for ticker, value in values.items()} if stock_value > 0 else {}


def strategy_weights(
    strategy: str,
    day_df: pd.DataFrame,
    panel: pd.DataFrame,
    prepared_a5_panel: pd.DataFrame,
    state: dict[str, Any],
    top_k: int,
    decision_step: int,
    regime_info: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    day_df = apply_regime_market_score(day_df, regime_info)
    valid = get_valid_universe(day_df)
    if valid.empty:
        return {}, {"fallback_used": True, "fallback_reason": "empty_valid_universe"}
    if strategy == "A0":
        weights, fallback = generate_equal_weight(day_df)
        return normalize_weights(weights), {"fallback_used": bool(fallback)}
    if strategy == "A1":
        weights, fallback = generate_rule_based_weight(day_df, top_k)
        return normalize_weights(weights), {"fallback_used": bool(fallback)}
    current = current_close_weights(state, day_df)
    id_mapping = create_id_mapping(panel)
    if strategy == "A2":
        variant = os.getenv(
            "PILOT_A2_VARIANT",
            str((BASE_DIR / "A2_VARIANT").read_text().strip() if (BASE_DIR / "A2_VARIANT").exists() else "A2b"),
        )
        generator = generate_A2a_weight if variant.lower() == "a2a" else generate_A2b_weight
        decision = generator(day_df, current, top_k=top_k, decision_step=decision_step, id_mapping=id_mapping)
        return normalize_weights(decision.weights), {"fallback_used": decision.fallback_used, "a2_variant": variant, **decision.log}
    if strategy == "A3":
        decision = generate_A3_weight(day_df, current, top_k=top_k, decision_step=decision_step, id_mapping=id_mapping)
        return normalize_weights(decision.weights), {"fallback_used": decision.fallback_used, **decision.log}
    if strategy == "A4":
        decision = generate_A4_weight(day_df, current, top_k=top_k, decision_step=decision_step, id_mapping=id_mapping)
        return normalize_weights(decision.weights), {"fallback_used": decision.fallback_used, **decision.log}
    if strategy == "A5":
        decision_date = day_df["date"].iloc[0]
        day_a5 = make_state_key(bucketize_scores(day_df))
        history = get_bayesian_history(prepared_a5_panel, decision_date)
        weights, fallback = generate_A5_weight(day_a5, history, top_k)
        return normalize_weights(weights), {"fallback_used": bool(fallback), "history_rows": int(len(history))}
    raise ValueError(strategy)


def default_state(strategy: str, initial_nav: float) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "cash": float(initial_nav),
        "qty": {},
        "nav_close": float(initial_nav),
        "last_marked_date": None,
        "pending_execution_date": None,
        "pending_weights": {},
    }


def load_state(path: Path, strategy: str, initial_nav: float) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        state["qty"] = {str(k): float(v) for k, v in state.get("qty", {}).items()}
        state["pending_weights"] = {str(k): float(v) for k, v in state.get("pending_weights", {}).items()}
        return state
    return default_state(strategy, initial_nav)


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(state), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_daily(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    if "date" in new.columns:
        new = new.drop_duplicates(["date"], keep="last")
    new.to_csv(path, index=False, encoding="utf-8-sig")


def append_trade_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    keys = [col for col in ["execution_date", "ticker"] if col in new.columns]
    if len(keys) == 2:
        new = new.drop_duplicates(keys, keep="last")
    new.to_csv(path, index=False, encoding="utf-8-sig")


def compute_performance_metrics(daily: pd.DataFrame, initial_nav: float) -> dict[str, float]:
    if daily.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "total_commission": 0.0,
            "commission_drag": 0.0,
            "total_turnover_value": 0.0,
            "avg_turnover_ratio": 0.0,
        }
    returns = daily["daily_return"].astype(float)
    nav = daily["nav_close"].astype(float)
    final_nav = float(nav.iloc[-1])
    years = max(len(daily) / 252.0, 1.0 / 252.0)
    total_return = final_nav / initial_nav - 1.0
    cagr = (final_nav / initial_nav) ** (1.0 / years) - 1.0
    annual_vol = float(returns.std(ddof=0) * math.sqrt(252))
    sharpe = float((returns.mean() * 252) / annual_vol) if annual_vol > 0 else 0.0
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0) * math.sqrt(252)) if len(downside) else 0.0
    sortino = float((returns.mean() * 252) / downside_vol) if downside_vol > 0 else 0.0
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    total_commission = float(daily["commission"].sum())
    total_turnover = float(daily["turnover_value"].sum())
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0,
        "total_commission": total_commission,
        "commission_drag": float(total_commission / initial_nav),
        "total_turnover_value": total_turnover,
        "avg_turnover_ratio": float(daily["turnover_ratio"].mean()),
    }


def rebalance_or_hold_one_day(
    state: dict[str, Any],
    day_df: pd.DataFrame,
    target_weights: dict[str, float] | None,
    stock_weight: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    date_text = pd.Timestamp(day_df["date"].iloc[0]).date().isoformat()
    qty = {str(k): float(v) for k, v in state.get("qty", {}).items()}
    cash = float(state.get("cash", 0.0))
    prev_nav = float(state.get("nav_close", INITIAL_NAV))
    open_px = dict(zip(day_df["ticker"].astype(str), day_df["open"].astype(float)))
    close_px = dict(zip(day_df["ticker"].astype(str), day_df["close"].astype(float)))
    universe = set(qty) | set(target_weights or {})
    old_values = {ticker: qty.get(ticker, 0.0) * open_px[ticker] for ticker in universe if ticker in open_px}
    nav_open_pre = cash + sum(old_values.values())
    commission = 0.0
    turnover_value = 0.0
    rebalanced = target_weights is not None
    if target_weights is not None:
        stock_ratio = float(stock_weight if stock_weight is not None else STOCK_RATIO)
        weights = {ticker: float(weight) for ticker, weight in target_weights.items() if weight > 0 and ticker in open_px}
        total = sum(weights.values())
        weights = {ticker: value / total for ticker, value in weights.items()} if total > 0 else {}
        nav_open_post = nav_open_pre
        target_values: dict[str, float] = {}
        target_qty: dict[str, float] = {}
        for _ in range(50):
            target_budget = {ticker: stock_ratio * nav_open_post * weight for ticker, weight in weights.items()}
            target_qty = {
                ticker: float(math.floor(value / open_px[ticker]))
                for ticker, value in target_budget.items()
                if ticker in open_px and open_px[ticker] > 0 and value > 0
            }
            target_values = {ticker: amount * open_px[ticker] for ticker, amount in target_qty.items()}
            names = set(target_values) | set(old_values)
            next_turnover = sum(abs(target_values.get(ticker, 0.0) - old_values.get(ticker, 0.0)) for ticker in names)
            next_commission = COMMISSION_RATE * next_turnover
            next_nav = nav_open_pre - next_commission
            if abs(next_nav - nav_open_post) <= 1e-10 * max(1.0, nav_open_pre):
                nav_open_post = next_nav
                commission = next_commission
                turnover_value = next_turnover
                break
            nav_open_post = next_nav
            commission = next_commission
            turnover_value = next_turnover
        target_budget = {ticker: stock_ratio * nav_open_post * weight for ticker, weight in weights.items()}
        target_qty = {
            ticker: float(math.floor(value / open_px[ticker]))
            for ticker, value in target_budget.items()
            if ticker in open_px and open_px[ticker] > 0 and value > 0
        }
        target_values = {ticker: amount * open_px[ticker] for ticker, amount in target_qty.items()}
        qty = target_qty
        cash = nav_open_post - sum(target_values.values())
    else:
        nav_open_post = nav_open_pre
    nav_close = cash + sum(amount * close_px.get(ticker, 0.0) for ticker, amount in qty.items())
    state.update({"cash": cash, "qty": qty, "nav_close": nav_close, "last_marked_date": date_text})
    result = {
        "date": date_text,
        "nav_open_pre": nav_open_pre,
        "nav_open_post": nav_open_post,
        "nav_close": nav_close,
        "daily_return": nav_close / prev_nav - 1.0 if prev_nav > 0 else 0.0,
        "commission": commission,
        "turnover_value": turnover_value,
        "turnover_ratio": turnover_value / nav_open_pre if nav_open_pre > 0 else 0.0,
        "rebalanced": rebalanced,
        "stock_weight": float(stock_weight) if stock_weight is not None else None,
        "cash_weight": float(1.0 - stock_weight) if stock_weight is not None else None,
        "effective_num_positions": sum(1 for value in qty.values() if abs(value) > 1e-12),
    }
    return state, result


def make_order_rows(
    strategy: str,
    state: dict[str, Any],
    decision_df: pd.DataFrame,
    target_weights: dict[str, float],
    execution_date: str,
    regime_info: dict[str, Any],
) -> list[dict[str, Any]]:
    close_px = dict(zip(decision_df["ticker"].astype(str), decision_df["close"].astype(float)))
    names = dict(zip(decision_df["ticker"].astype(str), decision_df["name"].astype(str)))
    qty = {str(k): float(v) for k, v in state.get("qty", {}).items()}
    nav = float(state.get("nav_close", 0.0))
    stock_weight = float(regime_info["stock_weight"])
    rows = []
    tickers = sorted(set(qty) | set(target_weights))
    current_values = {ticker: qty.get(ticker, 0.0) * close_px.get(ticker, 0.0) for ticker in tickers}
    for ticker in tickers:
        current_value = current_values.get(ticker, 0.0)
        current_portfolio_weight = current_value / nav if nav > 0 else 0.0
        target_sleeve_weight = float(target_weights.get(ticker, 0.0))
        target_portfolio_weight = stock_weight * target_sleeve_weight
        target_value = nav * target_portfolio_weight
        delta_value = target_value - current_value
        close = close_px.get(ticker, math.nan)
        current_qty = qty.get(ticker, 0.0)
        target_qty_at_close = math.floor(target_value / close) if close and close > 0 and math.isfinite(close) else 0
        executable_qty_change = target_qty_at_close - math.floor(current_qty)
        rows.append(
            {
                "strategy": strategy,
                "decision_date": pd.Timestamp(decision_df["date"].iloc[0]).date().isoformat(),
                "execution_date": execution_date,
                "ticker": ticker,
                "name": names.get(ticker, ticker),
                **regime_info,
                "current_qty": current_qty,
                "current_close": close,
                "current_portfolio_weight_at_close": current_portfolio_weight,
                "target_sleeve_weight": target_sleeve_weight,
                "target_portfolio_weight": target_portfolio_weight,
                "delta_portfolio_weight": target_portfolio_weight - current_portfolio_weight,
                "target_value_at_close": target_value,
                "delta_value_at_close": delta_value,
                "indicative_qty_change_at_close": delta_value / close if close and close > 0 else 0.0,
                "target_qty_at_close_floor": target_qty_at_close,
                "executable_qty_change_at_close": executable_qty_change,
                "side": "BUY" if executable_qty_change > 0 else ("SELL" if executable_qty_change < 0 else "HOLD"),
            }
        )
    return rows


def run_strategy(
    strategy: str,
    panel: pd.DataFrame,
    prepared_a5_panel: pd.DataFrame,
    start_date: str,
    as_of_date: str,
    next_open_date: str,
    top_k: int,
    initial_nav: float,
    out_dir: Path,
    regime: pd.DataFrame,
) -> dict[str, Any]:
    strategy_dir = out_dir / strategy
    state_path = strategy_dir / "state.json"
    state = load_state(state_path, strategy, initial_nav)
    daily_path = strategy_dir / "daily_pnl.csv"
    trade_path = strategy_dir / "trade_history.csv"
    marked = set(load_daily(daily_path).get("date", pd.Series(dtype=str)).astype(str))

    dates = [pd.Timestamp(d) for d in sorted(panel["date"].drop_duplicates())]
    start_ts = pd.Timestamp(start_date)
    as_of_ts = pd.Timestamp(as_of_date)
    if state.get("last_marked_date"):
        mark_dates = [d for d in dates if pd.Timestamp(state["last_marked_date"]) < d <= as_of_ts]
    else:
        prior_dates = [d for d in dates if d < start_ts]
        if not prior_dates:
            raise ValueError("Need at least one decision date before pilot start date for initial weights.")
        initial_decision = prior_dates[-1]
        day_df = panel.loc[panel["date"] == initial_decision]
        initial_regime = regime_for_date(regime, initial_decision.date().isoformat())
        weights, log = strategy_weights(
            strategy,
            day_df,
            panel,
            prepared_a5_panel,
            state,
            top_k,
            0,
            initial_regime,
        )
        state["pending_execution_date"] = start_ts.date().isoformat()
        state["pending_decision_date"] = initial_decision.date().isoformat()
        state["pending_weights"] = weights
        state["pending_stock_weight"] = float(initial_regime["stock_weight"])
        state["pending_regime"] = initial_regime
        state["last_decision_log"] = log
        mark_dates = [d for d in dates if start_ts <= d <= as_of_ts]

    daily_rows = []
    trade_rows: list[dict[str, Any]] = []
    for d in mark_dates:
        date_text = d.date().isoformat()
        if date_text in marked:
            continue
        day_df = panel.loc[panel["date"] == d]
        target = None
        stock_weight = None
        if state.get("pending_execution_date") == date_text:
            target = {str(k): float(v) for k, v in state.get("pending_weights", {}).items()}
            stock_weight = float(state.get("pending_stock_weight", STOCK_RATIO))
        old_qty = {str(k): float(v) for k, v in state.get("qty", {}).items()}
        pending_decision_date = str(state.get("pending_decision_date") or date_text)
        state, result = rebalance_or_hold_one_day(state, day_df, target, stock_weight)
        daily_rows.append(result)
        if target is not None:
            trade_rows.extend(
                build_trade_rows(
                    strategy=strategy,
                    decision_date=pending_decision_date,
                    execution_date=date_text,
                    decision_df=day_df,
                    execution_df=day_df,
                    old_qty=old_qty,
                    new_qty={str(k): float(v) for k, v in state.get("qty", {}).items()},
                    target_sleeve_weights=target,
                    result=result,
                    stock_ratio=float(stock_weight if stock_weight is not None else STOCK_RATIO),
                )
            )
        if d < as_of_ts:
            next_dates = [candidate for candidate in dates if candidate > d]
            if next_dates:
                next_execution_date = next_dates[0].date().isoformat()
                next_regime = regime_for_date(regime, date_text)
                next_weights, next_log = strategy_weights(
                    strategy,
                    day_df,
                    panel,
                    prepared_a5_panel,
                    state,
                    top_k,
                    len(marked) + len(daily_rows),
                    next_regime,
                )
                state["pending_execution_date"] = next_execution_date
                state["pending_decision_date"] = date_text
                state["pending_weights"] = next_weights
                state["pending_stock_weight"] = float(next_regime["stock_weight"])
                state["pending_regime"] = next_regime
                state["last_decision_date"] = date_text
                state["last_decision_log"] = next_log

    decision_df = panel.loc[panel["date"] == as_of_ts]
    if decision_df.empty:
        raise ValueError(f"No panel rows for as_of_date={as_of_date}")
    next_regime = regime_for_date(regime, as_of_date)
    weights, decision_log = strategy_weights(
        strategy,
        decision_df,
        panel,
        prepared_a5_panel,
        state,
        top_k,
        len(marked) + len(daily_rows),
        next_regime,
    )
    state["pending_execution_date"] = next_open_date
    state["pending_decision_date"] = as_of_date
    state["pending_weights"] = weights
    state["pending_stock_weight"] = float(next_regime["stock_weight"])
    state["pending_regime"] = next_regime
    state["last_decision_date"] = as_of_date
    state["last_decision_log"] = decision_log
    save_state(state_path, state)
    append_csv(daily_path, daily_rows)
    append_trade_csv(trade_path, trade_rows)
    daily_all = load_daily(daily_path)

    orders = make_order_rows(strategy, state, decision_df, weights, next_open_date, next_regime)
    orders_df = pd.DataFrame(orders)
    strategy_dir.mkdir(parents=True, exist_ok=True)
    orders_df.to_csv(strategy_dir / "next_orders.csv", index=False, encoding="utf-8-sig")
    if trade_path.exists():
        pd.read_csv(trade_path).to_csv(strategy_dir / "trade_history.csv", index=False, encoding="utf-8-sig")
    orders_df.to_csv(strategy_dir / "target_weights.csv", index=False, encoding="utf-8-sig")
    return {
        "strategy": strategy,
        "as_of_date": as_of_date,
        "next_open_date": next_open_date,
        "nav_close": float(state["nav_close"]),
        "pending_positions": len([v for v in weights.values() if v > 0]),
        "cash_weight": float(next_regime["cash_weight"]),
        "stock_weight": float(next_regime["stock_weight"]),
        "regime_key": next_regime["regime_key"],
        "fallback_used": bool(decision_log.get("fallback_used", False)),
        **compute_performance_metrics(daily_all, initial_nav),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A0-A5 pilot allocation and next-open order files.")
    parser.add_argument("--start-date", required=True, help="Pilot initial execution date, e.g. 2026-05-01")
    parser.add_argument("--as-of-date", required=True, help="Latest close data date, e.g. 2026-05-04")
    parser.add_argument("--next-open-date", required=True, help="Next execution date, e.g. 2026-05-06")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--initial-nav", type=float, default=INITIAL_NAV)
    parser.add_argument("--strategies", default=",".join(PILOT_STRATEGIES))
    parser.add_argument("--price-path", type=Path, default=BASE_DIR / "data" / "sector_all_merged.csv")
    parser.add_argument("--flow-path", type=Path, default=BASE_DIR / "data" / "sector_fund_flow.csv")
    parser.add_argument("--regime-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--fetch-from-date")
    return parser.parse_args()


def run_all_from_args(args: argparse.Namespace, only_strategy: str | None = None) -> None:
    load_dotenv()
    ensure_recent_data(args)
    price_path = read_csv_or_fallback(args.price_path, TESTING_DIR / "sector_all_merged.csv")
    flow_path = read_csv_or_fallback(args.flow_path, TESTING_DIR / "sector_fund_flow.csv")
    regime_path = args.regime_path or find_default_regime_path()
    regime = load_regime_table(regime_path)
    panel_path = build_rank_panel(price_path, flow_path)
    panel = load_panel(panel_path)
    prepared_a5_panel = prepare_A5_panel(apply_regime_market_score_by_next_execution(panel, regime))
    strategies = [only_strategy] if only_strategy else parse_strategy_list(args.strategies)
    summaries = []
    all_orders = []
    all_trades = []
    for strategy in strategies:
        summary = run_strategy(
            strategy=strategy,
            panel=panel,
            prepared_a5_panel=prepared_a5_panel,
            start_date=args.start_date,
            as_of_date=args.as_of_date,
            next_open_date=args.next_open_date,
            top_k=args.top_k,
            initial_nav=args.initial_nav,
            out_dir=args.out_dir,
            regime=regime,
        )
        summaries.append(summary)
        orders_path = args.out_dir / strategy / "next_orders.csv"
        if orders_path.exists():
            all_orders.append(pd.read_csv(orders_path))
        trades_path = args.out_dir / strategy / "trade_history.csv"
        if trades_path.exists():
            all_trades.append(pd.read_csv(trades_path))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.out_dir / "latest_summary.csv", index=False, encoding="utf-8-sig")
    if all_orders:
        combined = pd.concat(all_orders, ignore_index=True)
        combined.to_csv(args.out_dir / "latest_orders.csv", index=False, encoding="utf-8-sig")
        combined.to_csv(args.out_dir / "latest_target_weights.csv", index=False, encoding="utf-8-sig")
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(args.out_dir / "latest_trade_history.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main_single(strategy: str) -> None:
    args = parse_args()
    run_all_from_args(args, only_strategy=strategy)


def main() -> None:
    run_all_from_args(parse_args())


if __name__ == "__main__":
    main()

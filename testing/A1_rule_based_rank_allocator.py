from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from A0_equal_weight import (
    BacktestConfig,
    TOP_K_GRID,
    default_out_root,
    default_panel_path,
    get_valid_universe,
    load_rank_panel,
    normalize_sleeve_weights,
    run_backtest,
    validate_backtest_panel,
)


COMPOSITE_WEIGHTS = {
    "market_score": 0.30,
    "flow_score": 0.35,
    "rotation_score": 0.35,
}


def compute_composite_score(day_df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=day_df.index, dtype=float)
    for col, weight in COMPOSITE_WEIGHTS.items():
        score = score + weight * day_df[col].fillna(0.0).astype(float)
    return score.clip(lower=0.0)


def select_candidate_set(day_df: pd.DataFrame, score_col: str, top_k: int) -> pd.DataFrame:
    valid = get_valid_universe(day_df)
    if valid.empty:
        return valid
    candidates = valid.sort_values([score_col, "ticker"], ascending=[False, True])
    if top_k < 10:
        candidates = candidates.head(top_k)
    return candidates


def generate_rule_based_weight(day_df: pd.DataFrame, top_k: int) -> tuple[pd.Series, bool]:
    valid = get_valid_universe(day_df)
    if valid.empty:
        return pd.Series(dtype=float), True
    scored = valid.copy()
    scored["composite_score"] = compute_composite_score(scored)
    candidates = select_candidate_set(scored, "composite_score", top_k)
    raw = pd.Series(candidates["composite_score"].to_numpy(), index=candidates["ticker"].astype(str))
    return normalize_sleeve_weights(raw, candidates["ticker"].astype(str))


def run_A1_backtest(
    panel_path: Path | None = None,
    out_root: Path | None = None,
    top_k_values: tuple[int, ...] = TOP_K_GRID,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame, dict]]:
    panel_path = panel_path or default_panel_path()
    out_root = out_root or default_out_root()
    panel = load_rank_panel(panel_path)
    validate_backtest_panel(panel)
    results = {}
    for top_k in top_k_values:
        strategy = f"A1_RULE_BASED_RANK_ALLOCATOR_k{top_k}"
        results[top_k] = run_backtest(
            panel=panel,
            weight_generator=lambda day, k=top_k: generate_rule_based_weight(day, k),
            config=BacktestConfig(strategy=strategy),
            out_dir=out_root / f"A1_k{top_k}",
            panel_path=panel_path,
            extra_summary={"top_k": top_k, "composite_weights": COMPOSITE_WEIGHTS},
        )
    return results


def _parse_top_k(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A1 rule-based rank allocator backtests.")
    parser.add_argument("--panel", type=Path, default=default_panel_path())
    parser.add_argument("--out-root", type=Path, default=default_out_root())
    parser.add_argument("--top-k", type=_parse_top_k, default=TOP_K_GRID, help="Comma-separated grid, default 3,5,7,10.")
    args = parser.parse_args()
    run_A1_backtest(args.panel, args.out_root, args.top_k)


if __name__ == "__main__":
    main()

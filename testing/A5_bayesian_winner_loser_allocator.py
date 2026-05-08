from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from A0_equal_weight import (
    BacktestConfig,
    TOP_K_GRID,
    apply_regime_market_score_by_next_execution,
    default_out_root,
    default_panel_path,
    default_regime_path,
    get_valid_universe,
    load_regime_table,
    load_rank_panel,
    normalize_sleeve_weights,
    run_backtest,
    validate_backtest_panel,
)


ALPHA0 = 1.0
BETA0 = 1.0
SHRINK_K = 20.0
BUCKET_COLUMNS = ("market_bucket", "flow_bucket", "rotation_bucket")


def compute_internal_next_cc_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()
    panel["etf_next_cc_return"] = panel.groupby("ticker")["close"].shift(-1) / panel["close"] - 1.0

    index_by_date = panel.sort_values("date").drop_duplicates("date").set_index("date")["index_close"].astype(float)
    index_next = index_by_date.shift(-1) / index_by_date - 1.0
    panel["index_next_cc_return"] = panel["date"].map(index_next)
    return panel


def make_benchmark_beat_labels(panel: pd.DataFrame) -> pd.DataFrame:
    panel = compute_internal_next_cc_returns(panel)
    valid_label = panel["etf_next_cc_return"].notna() & panel["index_next_cc_return"].notna()
    panel["benchmark_beat_label"] = np.where(
        valid_label,
        (panel["etf_next_cc_return"] > panel["index_next_cc_return"]).astype(int),
        np.nan,
    )
    return panel


def _bucket(score: float) -> str:
    if pd.isna(score):
        return "missing"
    if score >= 0.67:
        return "top"
    if score >= 0.33:
        return "middle"
    return "bottom"


def bucketize_scores(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["market_bucket"] = panel["market_score"].map(_bucket)
    panel["flow_bucket"] = panel["flow_score"].map(_bucket)
    panel["rotation_bucket"] = panel["rotation_score"].map(_bucket)
    return panel


def make_state_key(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["state_key"] = (
        "M_"
        + panel["market_bucket"].astype(str)
        + "__F_"
        + panel["flow_bucket"].astype(str)
        + "__R_"
        + panel["rotation_bucket"].astype(str)
    )
    return panel


def prepare_A5_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return make_state_key(bucketize_scores(make_benchmark_beat_labels(panel)))


def get_bayesian_history(panel: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    history = panel.loc[
        (panel["date"] < decision_date)
        & panel["benchmark_beat_label"].notna()
        & panel["state_key"].notna()
    ].copy()
    return history


def fit_state_posteriors(
    history: pd.DataFrame,
    alpha0: float = ALPHA0,
    beta0: float = BETA0,
    shrink_k: float = SHRINK_K,
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(
            columns=[
                "state_key",
                "wins",
                "losses",
                "n",
                "posterior_mean",
                "global_posterior_mean",
                "shrink_lambda",
                "posterior_score",
            ]
        )

    grouped = history.groupby("state_key")["benchmark_beat_label"].agg(["sum", "count"]).reset_index()
    grouped = grouped.rename(columns={"sum": "wins", "count": "n"})
    grouped["losses"] = grouped["n"] - grouped["wins"]
    grouped["posterior_mean"] = (alpha0 + grouped["wins"]) / (alpha0 + beta0 + grouped["n"])

    global_wins = float(history["benchmark_beat_label"].sum())
    global_n = float(history["benchmark_beat_label"].count())
    global_posterior = (alpha0 + global_wins) / (alpha0 + beta0 + global_n)
    grouped["global_posterior_mean"] = global_posterior
    grouped["shrink_lambda"] = grouped["n"] / (grouped["n"] + shrink_k)
    grouped["posterior_score"] = (
        grouped["shrink_lambda"] * grouped["posterior_mean"]
        + (1.0 - grouped["shrink_lambda"]) * grouped["global_posterior_mean"]
    )
    return grouped


def score_current_day_with_posteriors(day_df: pd.DataFrame, posterior_table: pd.DataFrame) -> pd.DataFrame:
    scored = day_df.copy()
    if posterior_table.empty:
        scored["a5_posterior_score"] = 0.5
        return scored
    mapping = posterior_table.set_index("state_key")["posterior_score"]
    global_score = float(posterior_table["global_posterior_mean"].iloc[0])
    scored["a5_posterior_score"] = scored["state_key"].map(mapping).fillna(global_score).astype(float)
    return scored


def generate_A5_weight(
    day_df: pd.DataFrame,
    history_df: pd.DataFrame,
    top_k: int,
    posterior_sink: list[pd.DataFrame] | None = None,
) -> tuple[pd.Series, bool]:
    valid = get_valid_universe(day_df)
    if valid.empty:
        return pd.Series(dtype=float), True
    posterior_table = fit_state_posteriors(history_df)
    if posterior_sink is not None:
        snapshot = posterior_table.copy()
        snapshot.insert(0, "decision_date", day_df["date"].iloc[0].date().isoformat())
        posterior_sink.append(snapshot)
    scored = score_current_day_with_posteriors(valid, posterior_table)
    scored["a5_posterior_score"] = scored["a5_posterior_score"].clip(lower=0.0)
    scored = scored.sort_values(["a5_posterior_score", "ticker"], ascending=[False, True])
    if top_k < 10:
        scored = scored.head(top_k)
    raw = pd.Series(scored["a5_posterior_score"].to_numpy(), index=scored["ticker"].astype(str))
    return normalize_sleeve_weights(raw, scored["ticker"].astype(str))


def run_A5_backtest(
    panel_path: Path | None = None,
    out_root: Path | None = None,
    top_k_values: tuple[int, ...] = TOP_K_GRID,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame, dict]]:
    panel_path = panel_path or default_panel_path()
    out_root = out_root or default_out_root()
    raw_panel = load_rank_panel(panel_path)
    validate_backtest_panel(raw_panel, require_index_close=True)
    regime = load_regime_table(default_regime_path())
    panel = prepare_A5_panel(apply_regime_market_score_by_next_execution(raw_panel, regime))
    results = {}
    for top_k in top_k_values:
        posterior_rows: list[pd.DataFrame] = []

        def weight_generator(day: pd.DataFrame, k: int = top_k) -> tuple[pd.Series, bool]:
            decision_date = day["date"].iloc[0]
            history = get_bayesian_history(panel, decision_date)
            return generate_A5_weight(day, history, k, posterior_rows)

        strategy = f"A5_BAYESIAN_WINNER_LOSER_ALLOCATOR_k{top_k}"
        out_dir = out_root / f"A5_k{top_k}"
        results[top_k] = run_backtest(
            panel=panel,
            weight_generator=weight_generator,
            config=BacktestConfig(strategy=strategy),
            out_dir=out_dir,
            panel_path=panel_path,
            extra_summary={
                "top_k": top_k,
                "alpha0": ALPHA0,
                "beta0": BETA0,
                "shrink_k": SHRINK_K,
                "label": "ETF next close-to-close return > index next close-to-close return",
            },
            start_date=start_date,
            end_date=end_date,
        )
        posterior_states = pd.concat(posterior_rows, ignore_index=True) if posterior_rows else pd.DataFrame()
        posterior_states.to_csv(out_dir / "posterior_states.csv", index=False)
    return results


def _parse_top_k(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A5 Bayesian benchmark-beat allocator backtests.")
    parser.add_argument("--panel", type=Path, default=default_panel_path())
    parser.add_argument("--out-root", type=Path, default=default_out_root())
    parser.add_argument("--top-k", type=_parse_top_k, default=TOP_K_GRID, help="Comma-separated grid, default 3,5,7,10.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()
    run_A5_backtest(args.panel, args.out_root, args.top_k, args.start_date, args.end_date)


if __name__ == "__main__":
    main()

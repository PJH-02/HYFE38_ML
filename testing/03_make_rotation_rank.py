from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "name",
    "gics_sector",
    "close",
    "open",
    "volume",
    "amount",
}

OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "name",
    "gics_sector",
    "ret_1d",
    "sector_rs_4w",
    "sector_rs_rank_4w",
    "leader_sector",
    "leader_ticker",
    "leader_persistence",
    "leader_valid",
    "leader_changed",
    "transition_pair",
    "transition_frequency",
    "leader_persistence_score",
    "transition_frequency_score",
    "rotation_signal_week",
    "rotation_score_raw",
    "rotation_score",
    "rotation_rank",
    "rotation_valid",
    "rotation_obs",
]

EPS = 1.0e-12
LEADER_RS_THRESHOLD = 0.08
MAX_PERSISTENCE_SCORE_WEEKS = 4


def load_price_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "index_close" not in df.columns and "kospi_close" in df.columns:
        df = df.rename(columns={"kospi_close": "index_close"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def validate_rotation_schema(df: pd.DataFrame) -> None:
    missing = sorted((REQUIRED_COLUMNS | {"index_close"}) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date,ticker rows found")
    for col in ["open", "close", "volume", "amount", "index_close"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"{col} contains non-numeric or missing values")
    if (df[["open", "close", "index_close"]] <= 0).any().any():
        raise ValueError("open, close, and index_close must be positive")
    if (df[["volume", "amount"]] < 0).any().any():
        raise ValueError("volume and amount cannot be negative")


def compute_close_to_close_return(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    return out


def add_cross_sectional_zscore(df: pd.DataFrame, group_col: str, value_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    mean = out.groupby(group_col)[value_col].transform("mean")
    std = out.groupby(group_col)[value_col].transform("std")
    out[out_col] = (out[value_col] - mean) / (std + EPS)
    out[out_col] = out[out_col].where(std > EPS, 0.0)
    return out


def make_weekly_observations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rotation_week_period"] = out["date"].dt.to_period("W-FRI")
    weekly = (
        out.sort_values(["ticker", "date"])
        .groupby(["ticker", "rotation_week_period"], group_keys=False)
        .tail(1)
        .copy()
    )
    weekly["rotation_signal_week"] = weekly["rotation_week_period"].astype(str)
    weekly["rotation_signal_date"] = weekly["date"]
    weekly["etf_4w_log_return"] = weekly.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s / s.shift(4))
    )
    weekly["index_4w_log_return"] = weekly.groupby("ticker", sort=False)["index_close"].transform(
        lambda s: np.log(s / s.shift(4))
    )
    weekly["sector_rs_4w"] = weekly["etf_4w_log_return"] - weekly["index_4w_log_return"]
    weekly["sector_rs_rank_4w"] = np.nan
    valid_rs = np.isfinite(weekly["sector_rs_4w"])
    weekly.loc[valid_rs, "sector_rs_rank_4w"] = weekly.loc[valid_rs].groupby("rotation_week_period")[
        "sector_rs_4w"
    ].rank(method="first", ascending=False)
    return weekly


def build_leader_table(weekly: pd.DataFrame) -> pd.DataFrame:
    leader_rows = []
    valid = weekly.loc[weekly["sector_rs_rank_4w"].notna()].copy()
    for period, group in valid.groupby("rotation_week_period", sort=True):
        leader = group.sort_values(["sector_rs_4w", "ticker"], ascending=[False, True]).iloc[0]
        leader_rows.append(
            {
                "rotation_week_period": period,
                "leader_sector": str(leader["gics_sector"]),
                "leader_ticker": str(leader["ticker"]),
                "leader_sector_rs_4w": float(leader["sector_rs_4w"]),
            }
        )
    leaders = pd.DataFrame(leader_rows)
    if leaders.empty:
        return leaders

    leaders["prev_leader_sector"] = leaders["leader_sector"].shift(1)
    leaders["prev_leader_ticker"] = leaders["leader_ticker"].shift(1)
    leaders["leader_changed"] = leaders["prev_leader_ticker"].notna() & leaders["leader_ticker"].ne(
        leaders["prev_leader_ticker"]
    )
    leaders["leader_run_id"] = leaders["leader_ticker"].ne(leaders["prev_leader_ticker"]).cumsum()
    leaders["leader_persistence"] = leaders.groupby("leader_run_id").cumcount() + 1
    leaders["leader_valid"] = (leaders["leader_persistence"] >= 2) | (
        leaders["leader_sector_rs_4w"] >= LEADER_RS_THRESHOLD
    )
    leaders["transition_pair"] = ""
    changed = leaders["leader_changed"]
    leaders.loc[changed, "transition_pair"] = (
        leaders.loc[changed, "prev_leader_sector"].astype(str) + " -> " + leaders.loc[changed, "leader_sector"].astype(str)
    )

    counts: dict[str, int] = {}
    transition_frequency: list[int] = []
    transition_frequency_score: list[float] = []
    for row in leaders.itertuples(index=False):
        pair = str(row.transition_pair)
        use_transition = bool(row.leader_changed) and bool(row.leader_valid) and bool(pair)
        historical_count = counts.get(pair, 0) if use_transition else 0
        max_historical_count = max(counts.values()) if counts else 0
        score = historical_count / max_historical_count if use_transition and max_historical_count > 0 else 0.0
        transition_frequency.append(int(historical_count))
        transition_frequency_score.append(float(score))
        if use_transition:
            counts[pair] = historical_count + 1
    leaders["transition_frequency"] = transition_frequency
    leaders["transition_frequency_score_leader"] = transition_frequency_score
    return leaders


def compute_weekly_rotation_scores(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.copy()
    leaders = build_leader_table(out)
    if not leaders.empty:
        out = out.merge(
            leaders[
                [
                    "rotation_week_period",
                    "leader_sector",
                    "leader_ticker",
                    "leader_persistence",
                    "leader_valid",
                    "leader_changed",
                    "transition_pair",
                    "transition_frequency",
                    "transition_frequency_score_leader",
                ]
            ],
            on="rotation_week_period",
            how="left",
            validate="many_to_one",
        )
    else:
        out["leader_sector"] = pd.NA
        out["leader_ticker"] = pd.NA
        out["leader_persistence"] = pd.NA
        out["leader_valid"] = False
        out["leader_changed"] = False
        out["transition_pair"] = ""
        out["transition_frequency"] = 0
        out["transition_frequency_score_leader"] = 0.0

    out["leader_valid"] = out["leader_valid"].fillna(False).astype(bool)
    out["leader_changed"] = out["leader_changed"].fillna(False).astype(bool)
    out["transition_pair"] = out["transition_pair"].fillna("")
    out["transition_frequency"] = out["transition_frequency"].fillna(0).astype(int)
    out["is_leader"] = out["ticker"].astype(str).eq(out["leader_ticker"].astype(str))
    out["leader_persistence_score"] = np.where(
        out["is_leader"] & out["leader_valid"],
        np.minimum(out["leader_persistence"].fillna(0).astype(float), MAX_PERSISTENCE_SCORE_WEEKS)
        / MAX_PERSISTENCE_SCORE_WEEKS,
        0.0,
    )
    out["transition_frequency_score"] = np.where(
        out["is_leader"] & out["leader_changed"] & out["leader_valid"],
        out["transition_frequency_score_leader"].fillna(0.0).astype(float),
        0.0,
    )

    out = add_cross_sectional_zscore(out, "rotation_week_period", "sector_rs_4w", "sector_rs_4w_z")
    out["rotation_score_raw"] = (
        out["sector_rs_4w_z"].fillna(0.0).astype(float)
        + out["leader_persistence_score"].fillna(0.0).astype(float)
        + out["transition_frequency_score"].fillna(0.0).astype(float)
    ) / 3.0

    tradable = (
        (out["open"] > 0)
        & (out["close"] > 0)
        & (out["index_close"] > 0)
        & (out["volume"] > 0)
        & (out["amount"] > 0)
    )
    out["rotation_valid"] = tradable & np.isfinite(out["sector_rs_4w"]) & np.isfinite(out["rotation_score_raw"])
    out["rotation_rank"] = np.nan
    out["rotation_score"] = np.nan
    valid = out["rotation_valid"]
    out.loc[valid, "rotation_rank"] = out.loc[valid].groupby("rotation_week_period")[
        "rotation_score_raw"
    ].rank(method="first", ascending=False)
    n_by_week = out.loc[valid].groupby("rotation_week_period")["ticker"].transform("count")
    ranks = out.loc[valid, "rotation_rank"]
    out.loc[valid, "rotation_score"] = np.where(
        n_by_week > 1,
        1.0 - (ranks - 1.0) / (n_by_week - 1.0),
        1.0,
    )
    return out


def forward_fill_weekly_signal_to_daily(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    signal_columns = [
        "ticker",
        "rotation_signal_date",
        "rotation_signal_week",
        "sector_rs_4w",
        "sector_rs_rank_4w",
        "leader_sector",
        "leader_ticker",
        "leader_persistence",
        "leader_valid",
        "leader_changed",
        "transition_pair",
        "transition_frequency",
        "leader_persistence_score",
        "transition_frequency_score",
        "rotation_score_raw",
        "rotation_score",
        "rotation_rank",
        "rotation_valid",
    ]
    signals = weekly[signal_columns].copy()
    pieces = []
    ordered = daily.copy()
    ordered["__row_order"] = np.arange(len(ordered))
    for ticker, group in ordered.groupby("ticker", sort=False):
        ticker_signals = signals.loc[signals["ticker"].astype(str) == str(ticker)].drop(columns=["ticker"])
        if ticker_signals.empty:
            merged = group.copy()
            for col in signal_columns:
                if col != "ticker":
                    merged[col] = pd.NA
        else:
            merged = pd.merge_asof(
                group.sort_values("date"),
                ticker_signals.sort_values("rotation_signal_date"),
                left_on="date",
                right_on="rotation_signal_date",
                direction="backward",
                allow_exact_matches=True,
            )
        pieces.append(merged)
    out = pd.concat(pieces, ignore_index=True).sort_values("__row_order").drop(columns=["__row_order"])
    out["rotation_valid"] = out["rotation_valid"].fillna(False).astype(bool)
    out["leader_valid"] = out["leader_valid"].fillna(False).astype(bool)
    out["leader_changed"] = out["leader_changed"].fillna(False).astype(bool)
    out["transition_pair"] = out["transition_pair"].fillna("")
    out["transition_frequency"] = out["transition_frequency"].fillna(0).astype(int)
    out["rotation_obs"] = out.groupby("date")["rotation_valid"].transform("sum").astype(int)
    return out


def make_rotation_rank(df: pd.DataFrame) -> pd.DataFrame:
    daily = compute_close_to_close_return(df)
    weekly = compute_weekly_rotation_scores(make_weekly_observations(daily))
    out = forward_fill_weekly_signal_to_daily(daily, weekly)
    return out


def validate_no_lookahead_rotation(df: pd.DataFrame) -> None:
    forbidden = {"next_open", "next_close", "future_return", "next_return", "label"}
    leaked = sorted(forbidden & set(df.columns))
    if leaked:
        raise ValueError(f"Forbidden look-ahead columns present: {leaked}")
    valid_signal_date = df["rotation_signal_date"].notna()
    if (df.loc[valid_signal_date, "rotation_signal_date"] > df.loc[valid_signal_date, "date"]).any():
        raise ValueError("rotation signal date is after rank date")


def save_rotation_rank(df: pd.DataFrame, out_path: str | Path) -> None:
    out = df[OUTPUT_COLUMNS].copy()
    validate_no_lookahead_rotation(df)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(out_path, index=False)


def build_rotation_rank(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    df = load_price_data(input_path)
    validate_rotation_schema(df)
    df = make_rotation_rank(df)
    validate_no_lookahead_rotation(df)
    save_rotation_rank(df, output_path)
    return df


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create weekly relative-strength rotation rank CSV.")
    parser.add_argument("--input", default=base / "sector_all_merged.csv")
    parser.add_argument("--output", default=base / "rotation_rank.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_rotation_rank(input_path=args.input, output_path=args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

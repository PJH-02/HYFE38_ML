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
    "rotation_score_raw",
    "rotation_score",
    "rotation_rank",
    "rotation_valid",
    "rotation_obs",
]

EPS = 1.0e-12


def load_price_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def validate_rotation_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date,ticker rows found")
    for col in ["open", "close", "volume", "amount"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"{col} contains non-numeric or missing values")
    if (df[["open", "close"]] <= 0).any().any():
        raise ValueError("open and close must be positive")
    if (df[["volume", "amount"]] < 0).any().any():
        raise ValueError("volume and amount cannot be negative")


def compute_close_to_close_return(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    return out


def cross_sectional_zscore(
    df: pd.DataFrame, value_col: str, out_col: str
) -> pd.DataFrame:
    out = df.copy()
    mean = out.groupby("date")[value_col].transform("mean")
    std = out.groupby("date")[value_col].transform("std")
    out[out_col] = (out[value_col] - mean) / (std + EPS)
    out[out_col] = out[out_col].where(std > EPS, 0.0)
    return out


def compute_rotation_score(df: pd.DataFrame) -> pd.DataFrame:
    return cross_sectional_zscore(df, value_col="ret_1d", out_col="rotation_score_raw")


def make_rotation_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tradable = (
        (out["open"] > 0)
        & (out["close"] > 0)
        & (out["volume"] > 0)
        & (out["amount"] > 0)
    )
    out["rotation_valid"] = tradable & np.isfinite(out["ret_1d"]) & np.isfinite(out["rotation_score_raw"])
    out["rotation_obs"] = out.groupby("date")["rotation_valid"].transform("sum").astype(int)
    out["rotation_rank"] = np.nan
    out["rotation_score"] = np.nan

    valid = out["rotation_valid"]
    out.loc[valid, "rotation_rank"] = out.loc[valid].groupby("date")[
        "rotation_score_raw"
    ].rank(method="first", ascending=False)
    n_by_date = out.loc[valid].groupby("date")["ticker"].transform("count")
    ranks = out.loc[valid, "rotation_rank"]
    scores = np.where(n_by_date > 1, 1.0 - (ranks - 1.0) / (n_by_date - 1.0), 1.0)
    out.loc[valid, "rotation_score"] = scores
    return out


def validate_no_lookahead_rotation(df: pd.DataFrame) -> None:
    forbidden = {"next_open", "next_close", "future_return", "next_return", "label"}
    leaked = sorted(forbidden & set(df.columns))
    if leaked:
        raise ValueError(f"Forbidden look-ahead columns present: {leaked}")


def save_rotation_rank(df: pd.DataFrame, out_path: str | Path) -> None:
    out = df[OUTPUT_COLUMNS].copy()
    validate_no_lookahead_rotation(out)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(out_path, index=False)


def build_rotation_rank(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    df = load_price_data(input_path)
    validate_rotation_schema(df)
    df = compute_close_to_close_return(df)
    df = compute_rotation_score(df)
    df = make_rotation_rank(df)
    validate_no_lookahead_rotation(df)
    save_rotation_rank(df, output_path)
    return df


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create rotation rank CSV.")
    parser.add_argument("--input", default=base / "sector_all_merged.csv")
    parser.add_argument("--output", default=base / "rotation_rank.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_rotation_rank(input_path=args.input, output_path=args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

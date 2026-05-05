from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "name",
    "open",
    "close",
    "volume",
    "amount",
    "gics_sector",
    "kospi_close",
}

OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "name",
    "gics_sector",
    "market_corr_active",
    "market_corr_rolling20",
    "market_corr_ewma20",
    "market_beta_rolling20",
    "market_score",
    "market_rank",
    "market_valid",
    "market_corr_mode",
    "market_lookback",
    "market_obs",
]


def load_price_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.rename(columns={"kospi_close": "index_close"})
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def validate_price_schema(df: pd.DataFrame) -> None:
    required = (REQUIRED_COLUMNS - {"kospi_close"}) | {"index_close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date,ticker rows found")
    positive_cols = ["open", "close", "index_close", "volume", "amount"]
    for col in positive_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"{col} contains non-numeric or missing values")
    if (df[["open", "close", "index_close"]] <= 0).any().any():
        raise ValueError("open, close, and index_close must be positive")
    if (df[["volume", "amount"]] < 0).any().any():
        raise ValueError("volume and amount cannot be negative")


def compute_close_to_close_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["etf_cc_ret"] = out.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    out["index_cc_ret"] = out.groupby("ticker", sort=False)["index_close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    return out


def compute_rolling_corr_beta(
    df: pd.DataFrame, lookback: int, min_obs: int
) -> pd.DataFrame:
    out = df.copy()

    def per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        ret = g["etf_cc_ret"]
        idx = g["index_cc_ret"]
        obs = ret.notna().mul(idx.notna()).rolling(lookback).sum()
        corr = ret.rolling(lookback, min_periods=min_obs).corr(idx)
        cov = ret.rolling(lookback, min_periods=min_obs).cov(idx)
        var = idx.rolling(lookback, min_periods=min_obs).var()
        beta = cov / var.replace(0.0, np.nan)
        return pd.DataFrame(
            {
                "market_corr_rolling20": corr,
                "market_beta_rolling20": beta,
                "market_obs": obs,
            },
            index=g.index,
        )

    stats = out.groupby("ticker", group_keys=False, sort=False).apply(per_ticker)
    out[["market_corr_rolling20", "market_beta_rolling20", "market_obs"]] = stats
    out["market_obs"] = out["market_obs"].fillna(0).astype(int)
    return out


def compute_ewma_corr(df: pd.DataFrame, halflife: int, min_obs: int) -> pd.DataFrame:
    out = df.copy()

    def per_ticker(g: pd.DataFrame) -> pd.Series:
        x = g["etf_cc_ret"]
        y = g["index_cc_ret"]
        valid = x.notna() & y.notna()
        cov = x.ewm(halflife=halflife, min_periods=min_obs, adjust=False).cov(y)
        var_x = x.ewm(halflife=halflife, min_periods=min_obs, adjust=False).var()
        var_y = y.ewm(halflife=halflife, min_periods=min_obs, adjust=False).var()
        corr = cov / np.sqrt(var_x * var_y)
        corr = corr.where(valid.cumsum() >= min_obs)
        return corr

    out["market_corr_ewma20"] = (
        out.groupby("ticker", group_keys=False, sort=False).apply(per_ticker).reset_index(level=0, drop=True)
    )
    return out


def select_active_market_corr(df: pd.DataFrame, corr_mode: str) -> pd.DataFrame:
    out = df.copy()
    if corr_mode == "rolling":
        out["market_corr_active"] = out["market_corr_rolling20"]
    elif corr_mode == "ewma":
        out["market_corr_active"] = out["market_corr_ewma20"]
    else:
        raise ValueError("corr_mode must be 'rolling' or 'ewma'")
    out["market_corr_mode"] = corr_mode
    return out


def make_market_rank(df: pd.DataFrame, lookback: int, min_obs: int) -> pd.DataFrame:
    out = df.copy()
    tradable = (
        (out["open"] > 0)
        & (out["close"] > 0)
        & (out["volume"] > 0)
        & (out["amount"] > 0)
    )
    out["market_valid"] = (
        tradable
        & (out["market_obs"] >= min_obs)
        & np.isfinite(out["market_corr_active"])
    )
    out["market_rank"] = np.nan
    out["market_score"] = np.nan

    valid = out["market_valid"]
    out.loc[valid, "market_rank"] = out.loc[valid].groupby("date")[
        "market_corr_active"
    ].rank(method="first", ascending=True)

    n_by_date = out.loc[valid].groupby("date")["ticker"].transform("count")
    ranks = out.loc[valid, "market_rank"]
    scores = np.where(n_by_date > 1, 1.0 - (ranks - 1.0) / (n_by_date - 1.0), 1.0)
    out.loc[valid, "market_score"] = scores
    out["market_lookback"] = lookback
    return out


def validate_no_lookahead_market(df: pd.DataFrame, min_obs: int) -> None:
    forbidden = {"next_open", "next_close", "future_return", "next_return", "label"}
    leaked = sorted(forbidden & set(df.columns))
    if leaked:
        raise ValueError(f"Forbidden look-ahead columns present: {leaked}")
    early_valid = df["market_valid"] & (df["market_obs"] < min_obs)
    if early_valid.any():
        raise ValueError("Rows with insufficient observations marked market_valid=True")


def save_market_rank(df: pd.DataFrame, out_path: str | Path) -> None:
    out = df[OUTPUT_COLUMNS].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(out_path, index=False)


def build_market_rank(
    input_path: str | Path,
    output_path: str | Path,
    lookback: int = 20,
    min_obs: int = 20,
    ewma_halflife: int = 20,
    corr_mode: str = "rolling",
) -> pd.DataFrame:
    df = load_price_data(input_path)
    validate_price_schema(df)
    df = compute_close_to_close_returns(df)
    df = compute_rolling_corr_beta(df, lookback=lookback, min_obs=min_obs)
    df = compute_ewma_corr(df, halflife=ewma_halflife, min_obs=min_obs)
    df = select_active_market_corr(df, corr_mode=corr_mode)
    df = make_market_rank(df, lookback=lookback, min_obs=min_obs)
    validate_no_lookahead_market(df, min_obs=min_obs)
    save_market_rank(df, output_path)
    return df


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create market correlation rank CSV.")
    parser.add_argument("--input", default=base / "sector_all_merged.csv")
    parser.add_argument("--output", default=base / "market_rank.csv")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--min-obs", type=int, default=20)
    parser.add_argument("--ewma-halflife", type=int, default=20)
    parser.add_argument("--corr-mode", choices=["rolling", "ewma"], default="rolling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_market_rank(
        input_path=args.input,
        output_path=args.output,
        lookback=args.lookback,
        min_obs=args.min_obs,
        ewma_halflife=args.ewma_halflife,
        corr_mode=args.corr_mode,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

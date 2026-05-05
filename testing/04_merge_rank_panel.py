from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

PRICE_COLUMNS = [
    "date",
    "ticker",
    "name",
    "gics_sector",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "index_close",
]

MARKET_COLUMNS = [
    "market_corr_active",
    "market_corr_rolling20",
    "market_corr_ewma20",
    "market_beta_rolling20",
    "market_score",
    "market_rank",
    "market_valid",
]

FLOW_COLUMNS = [
    "flow_score_raw",
    "flow_score",
    "flow_rank",
    "flow_valid",
]

ROTATION_COLUMNS = [
    "rotation_score_raw",
    "rotation_score",
    "rotation_rank",
    "rotation_valid",
]

OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "ETF_id",
    "name",
    "gics_sector",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "index_close",
    *MARKET_COLUMNS,
    *FLOW_COLUMNS,
    *ROTATION_COLUMNS,
    "tradable",
    "all_rank_valid",
]

FORBIDDEN_COLUMN_PARTS = (
    "next_open",
    "next_close",
    "next_return",
    "future_return",
    "winner_label",
)


def load_panel_inputs(
    price_path: Path,
    market_path: Path,
    flow_path: Path,
    rotation_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(price_path, dtype={"ticker": "string"}),
        pd.read_csv(market_path, dtype={"ticker": "string"}),
        pd.read_csv(flow_path, dtype={"ticker": "string"}),
        pd.read_csv(rotation_path, dtype={"ticker": "string"}),
    )


def standardize_date_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or "ticker" not in out.columns:
        raise ValueError("input is missing required date/ticker columns")
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.strftime("%Y-%m-%d")
    out["ticker"] = out["ticker"].astype("string").str.strip()
    if out["ticker"].isna().any() or (out["ticker"] == "").any():
        raise ValueError("ticker contains missing or empty values")
    return out


def standardize_price_panel(price_df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_date_ticker(price_df)
    if "kospi_close" in df.columns and "index_close" not in df.columns:
        df = df.rename(columns={"kospi_close": "index_close"})

    missing = [col for col in PRICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"price input is missing columns: {missing}")

    duplicate_mask = df.duplicated(["date", "ticker"], keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, ["date", "ticker"]].head().to_dict("records")
        raise ValueError(f"price input has duplicate date/ticker rows: {examples}")

    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "index_close"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df = df[PRICE_COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def create_etf_id_mapping(price_df: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(price_df["ticker"].dropna().astype(str).unique())
    return pd.DataFrame(
        {
            "ticker": tickers,
            "ETF_id": [f"ETF_{idx:03d}" for idx in range(1, len(tickers) + 1)],
            "asset_id": [f"asset_{idx:03d}" for idx in range(1, len(tickers) + 1)],
        }
    )


def prepare_rank_frame(df: pd.DataFrame, rank_columns: list[str], label: str) -> pd.DataFrame:
    out = standardize_date_ticker(df)
    duplicate_mask = out.duplicated(["date", "ticker"], keep=False)
    if duplicate_mask.any():
        examples = out.loc[duplicate_mask, ["date", "ticker"]].head().to_dict("records")
        raise ValueError(f"{label} rank input has duplicate date/ticker rows: {examples}")

    keep_cols = ["date", "ticker", *[col for col in rank_columns if col in out.columns]]
    missing_rank = [col for col in rank_columns if col.endswith("_rank") and col not in out.columns]
    if missing_rank:
        raise ValueError(f"{label} rank input is missing columns: {missing_rank}")

    return out[keep_cols]


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
        }
    )
    parsed = parsed.where(normalized.notna(), False)
    if parsed.isna().any():
        bad_values = sorted(series.loc[parsed.isna()].dropna().astype(str).unique())
        raise ValueError(f"invalid boolean values: {bad_values}")
    return parsed.astype(bool)


def merge_rank_files(
    price_df: pd.DataFrame,
    market_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    rotation_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    panel = price_df.merge(mapping_df[["ticker", "ETF_id"]], on="ticker", how="left", validate="many_to_one")
    panel = panel.merge(
        prepare_rank_frame(market_df, MARKET_COLUMNS, "market"),
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        prepare_rank_frame(flow_df, FLOW_COLUMNS, "flow"),
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        prepare_rank_frame(rotation_df, ROTATION_COLUMNS, "rotation"),
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    return panel


def normalize_valid_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for prefix in ("market", "flow", "rotation"):
        rank_col = f"{prefix}_rank"
        valid_col = f"{prefix}_valid"
        if valid_col not in out.columns:
            out[valid_col] = out[rank_col].notna()
        else:
            out[valid_col] = parse_bool_series(out[valid_col])
    return out


def compute_rank_scores_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_valid_flags(df)
    for prefix in ("market", "flow", "rotation"):
        rank_col = f"{prefix}_rank"
        score_col = f"{prefix}_score"
        valid_col = f"{prefix}_valid"
        if score_col in out.columns:
            continue

        valid = out[valid_col] & out[rank_col].notna()
        counts = out.loc[valid].groupby("date")[rank_col].transform("count")
        scores = pd.Series(pd.NA, index=out.index, dtype="Float64")
        valid_idx = valid[valid].index
        scores.loc[valid_idx] = 1.0
        multi_idx = valid_idx[counts.to_numpy() > 1]
        scores.loc[multi_idx] = 1 - (out.loc[multi_idx, rank_col].astype(float) - 1) / (
            counts.loc[multi_idx].astype(float) - 1
        )
        out[score_col] = scores
    return out


def make_tradable_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tradable"] = (
        (out["open"] > 0)
        & (out["close"] > 0)
        & (out["volume"] > 0)
        & (out["amount"] > 0)
    )
    out["all_rank_valid"] = out["market_valid"] & out["flow_valid"] & out["rotation_valid"]
    return out


def validate_rank_panel(df: pd.DataFrame, mapping_df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(["date", "ticker"], keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, ["date", "ticker"]].head().to_dict("records")
        raise ValueError(f"rank panel has duplicate date/ticker rows: {examples}")

    for col in ("open", "close", "index_close"):
        if (df[col] <= 0).any():
            raise ValueError(f"rank panel contains non-positive {col}")

    missing = [col for col in ("market_rank", "flow_rank", "rotation_rank") if col not in df.columns]
    if missing:
        raise ValueError(f"rank panel is missing rank columns: {missing}")

    forbidden = [
        col
        for col in df.columns
        if any(part in col.lower() for part in FORBIDDEN_COLUMN_PARTS)
    ]
    if forbidden:
        raise ValueError(f"rank panel contains future/label columns: {forbidden}")

    mapped = df[["ticker", "ETF_id"]].drop_duplicates()
    if mapped["ETF_id"].isna().any():
        raise ValueError("rank panel contains rows without ETF_id")
    if mapped.duplicated("ticker").any() or mapped.duplicated("ETF_id").any():
        raise ValueError("ETF_id mapping is not one-to-one")
    if df["ticker"].astype(str).str.startswith("ETF_").any():
        raise ValueError("ticker appears to have been overwritten by ETF_id")

    expected_mapping_cols = ["ticker", "ETF_id", "asset_id"]
    if list(mapping_df.columns) != expected_mapping_cols:
        raise ValueError(f"id mapping columns must be {expected_mapping_cols}")


def save_rank_panel(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def build_rank_panel(
    price_path: Path,
    market_path: Path,
    flow_path: Path,
    rotation_path: Path,
    out_path: Path,
    mapping_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_df, market_df, flow_df, rotation_df = load_panel_inputs(
        price_path, market_path, flow_path, rotation_path
    )
    price_df = standardize_price_panel(price_df)
    mapping_df = create_etf_id_mapping(price_df)
    panel = merge_rank_files(price_df, market_df, flow_df, rotation_df, mapping_df)
    panel = compute_rank_scores_if_missing(panel)
    panel = make_tradable_flags(panel)

    for col in OUTPUT_COLUMNS:
        if col not in panel.columns:
            panel[col] = pd.NA
    panel = panel[OUTPUT_COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)

    validate_rank_panel(panel, mapping_df)
    save_rank_panel(panel, out_path)
    save_rank_panel(mapping_df, mapping_path)
    return panel, mapping_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ETF rank files into rank_panel.csv.")
    parser.add_argument("--price-path", type=Path, default=BASE_DIR / "sector_all_merged.csv")
    parser.add_argument("--market-path", type=Path, default=BASE_DIR / "market_rank.csv")
    parser.add_argument("--flow-path", type=Path, default=BASE_DIR / "flow_rank.csv")
    parser.add_argument("--rotation-path", type=Path, default=BASE_DIR / "rotation_rank.csv")
    parser.add_argument("--out-path", type=Path, default=BASE_DIR / "rank_panel.csv")
    parser.add_argument("--mapping-path", type=Path, default=BASE_DIR / "id_mapping.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel, mapping = build_rank_panel(
        args.price_path,
        args.market_path,
        args.flow_path,
        args.rotation_path,
        args.out_path,
        args.mapping_path,
    )
    print(f"wrote {args.out_path} ({len(panel):,} rows)")
    print(f"wrote {args.mapping_path} ({len(mapping):,} tickers)")


if __name__ == "__main__":
    main()

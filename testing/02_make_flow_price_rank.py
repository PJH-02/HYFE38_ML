from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


BASE_FLOW_ACTOR_COLUMNS = [
    "individual_net_buy",
    "foreign_net_buy",
    "institution_net_buy",
    "finance_invest_net_buy",
    "insurance_net_buy",
    "trust_net_buy",
    "other_finance_net_buy",
    "bank_net_buy",
    "pension_net_buy",
    "private_fund_net_buy",
    "nation_net_buy",
    "other_corp_net_buy",
    "other_foreign_net_buy",
]

DERIVED_FLOW_ACTOR_COLUMNS = ["total_net_buy"]
FLOW_ACTOR_COLUMNS = DERIVED_FLOW_ACTOR_COLUMNS + BASE_FLOW_ACTOR_COLUMNS
PRICE_COLUMNS = ["date", "ticker", "name", "gics_sector", "open", "close", "volume", "amount"]
EPS = 1.0e-12
ACTOR_SELECTION_MODES = {"dynamic", "team_static"}

TEAM_STATIC_FLOW_DIRECTIONS = {
    "Energy": {
        "total_net_buy": 1.0,
        "institution_net_buy": 1.0,
        "individual_net_buy": -1.0,
    },
    "Materials": {
        "total_net_buy": -1.0,
        "institution_net_buy": -1.0,
    },
    "Information Technology": {
        "total_net_buy": 1.0,
        "institution_net_buy": 1.0,
        "individual_net_buy": -1.0,
    },
    "Financials": {},
    "Industrials": {
        "foreign_net_buy": -1.0,
    },
    "Consumer Discretionary": {
        "individual_net_buy": 1.0,
    },
    "Consumer Staples": {
        "total_net_buy": 1.0,
        "institution_net_buy": 1.0,
        "individual_net_buy": -1.0,
    },
    "Health Care": {
        "total_net_buy": -1.0,
        "institution_net_buy": -1.0,
        "individual_net_buy": 1.0,
    },
    "Communication Services": {},
    "Real Estate": {
        "total_net_buy": 1.0,
        "foreign_net_buy": -1.0,
        "institution_net_buy": 1.0,
        "individual_net_buy": -1.0,
    },
}


def load_price_and_flow(
    price_path: str | Path, flow_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_df = pd.read_csv(price_path)
    flow_df = pd.read_csv(flow_path)
    for df in (price_df, flow_df):
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str)
    validate_flow_schema(price_df, flow_df)
    flow_df = add_derived_flow_columns(flow_df)

    price_cols = PRICE_COLUMNS
    flow_cols = ["date", "ticker"] + FLOW_ACTOR_COLUMNS
    merged = price_df[price_cols].merge(
        flow_df[flow_cols], on=["date", "ticker"], how="inner", validate="one_to_one"
    )
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    return price_df, flow_df, merged


def validate_flow_schema(price_df: pd.DataFrame, flow_df: pd.DataFrame) -> None:
    price_required = set(PRICE_COLUMNS)
    flow_required = {"date", "ticker"} | set(BASE_FLOW_ACTOR_COLUMNS)
    price_missing = sorted(price_required - set(price_df.columns))
    flow_missing = sorted(flow_required - set(flow_df.columns))
    if price_missing:
        raise ValueError(f"Missing price columns: {price_missing}")
    if flow_missing:
        raise ValueError(f"Missing flow columns: {flow_missing}")
    if price_df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date,ticker rows found in price data")
    if flow_df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date,ticker rows found in flow data")

    for col in ["open", "close", "volume", "amount"]:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
    if price_df[["open", "close", "volume", "amount"]].isna().any().any():
        raise ValueError("Price data contains non-numeric values")
    if (price_df[["open", "close"]] <= 0).any().any():
        raise ValueError("open and close must be positive")
    if (price_df[["volume", "amount"]] < 0).any().any():
        raise ValueError("volume and amount cannot be negative")

    for col in BASE_FLOW_ACTOR_COLUMNS:
        flow_df[col] = pd.to_numeric(flow_df[col], errors="coerce")
    if flow_df[BASE_FLOW_ACTOR_COLUMNS].isna().any().any():
        raise ValueError("Flow actor columns contain non-numeric values")


def add_derived_flow_columns(flow_df: pd.DataFrame) -> pd.DataFrame:
    out = flow_df.copy()
    if "총 FF" in out.columns:
        out["total_net_buy"] = pd.to_numeric(out["총 FF"], errors="coerce").fillna(0.0)
    elif "total_net_buy" in out.columns:
        out["total_net_buy"] = pd.to_numeric(out["total_net_buy"], errors="coerce").fillna(0.0)
    else:
        out["total_net_buy"] = (
            out["foreign_net_buy"].astype(float)
            + out["institution_net_buy"].astype(float)
        )
    return out


def compute_internal_next_open_to_close_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["next_oc_ret"] = out.groupby("ticker", sort=False).apply(
        lambda g: np.log(g["close"].shift(-1) / g["open"].shift(-1))
    ).reset_index(level=0, drop=True)
    return out


def normalize_actor_flows(df: pd.DataFrame, actor_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    denominator = out["amount"] * 1000.0
    for actor in actor_cols:
        ratio = pd.Series(0.0, index=out.index, dtype=float)
        valid_amount = denominator > 0
        ratio.loc[valid_amount] = out.loc[valid_amount, actor].astype(float) / denominator.loc[valid_amount]
        out[f"{actor}_flow_ratio"] = ratio
    return out


def sanity_check_flow_units(df: pd.DataFrame, actor_cols: list[str]) -> None:
    ratio_cols = [f"{actor}_flow_ratio" for actor in actor_cols]
    q99 = df[ratio_cols].abs().quantile(0.99).max()
    if pd.notna(q99) and q99 > 10:
        warnings.warn(
            f"Large flow_ratio 99th percentile detected ({q99:.4g}); check input units.",
            RuntimeWarning,
            stacklevel=2,
        )


def compute_rolling_flow_zscore(
    df: pd.DataFrame, actor_cols: list[str], lookback: int, min_obs: int
) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("ticker", sort=False)
    for actor in actor_cols:
        ratio_col = f"{actor}_flow_ratio"
        mean = grouped[ratio_col].transform(
            lambda s: s.rolling(lookback, min_periods=min_obs).mean().shift(1)
        )
        std = grouped[ratio_col].transform(
            lambda s: s.rolling(lookback, min_periods=min_obs).std().shift(1)
        )
        z = (out[ratio_col] - mean) / (std + EPS)
        out[f"{actor}_flow_z"] = z.where(std > EPS, 0.0)
    return out


def compute_predictive_actor_corr(
    df: pd.DataFrame, actor_cols: list[str], lookback: int, min_obs: int
) -> pd.DataFrame:
    out = df.copy()

    for actor in actor_cols:
        z_col = f"{actor}_flow_z"
        corr_col = f"{actor}_predictive_corr"
        obs_col = f"{actor}_predictive_obs"

        def per_ticker(g: pd.DataFrame) -> pd.DataFrame:
            valid_pair = g[z_col].notna() & g["next_oc_ret"].notna()
            obs = valid_pair.rolling(lookback).sum().shift(1)
            corr = (
                g[z_col]
                .rolling(lookback, min_periods=min_obs)
                .corr(g["next_oc_ret"])
                .shift(1)
            )
            return pd.DataFrame({corr_col: corr, obs_col: obs}, index=g.index)

        stats = out.groupby("ticker", group_keys=False, sort=False).apply(per_ticker)
        out[[corr_col, obs_col]] = stats
        out[obs_col] = out[obs_col].fillna(0).astype(int)
    return out


def shrink_predictive_corr(df: pd.DataFrame, actor_cols: list[str], shrink_k: float) -> pd.DataFrame:
    out = df.copy()
    for actor in actor_cols:
        corr_col = f"{actor}_predictive_corr"
        obs_col = f"{actor}_predictive_obs"
        shrunk_col = f"{actor}_predictive_corr_shrunk"
        obs = out[obs_col].astype(float)
        out[shrunk_col] = out[corr_col].fillna(0.0) * (obs / (obs + shrink_k))
    return out


def select_predictive_actors(
    df: pd.DataFrame, actor_cols: list[str], corr_threshold: float
) -> pd.DataFrame:
    out = df.copy()
    for actor in actor_cols:
        shrunk_col = f"{actor}_predictive_corr_shrunk"
        selected_col = f"{actor}_selected"
        direction_col = f"{actor}_direction"
        corr = out[shrunk_col].astype(float).replace([np.inf, -np.inf], np.nan)
        out[selected_col] = corr.abs() >= corr_threshold
        out[direction_col] = np.sign(corr).where(out[selected_col], 0.0)
    return out


def select_team_static_actors(df: pd.DataFrame, actor_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for actor in actor_cols:
        out[f"{actor}_selected"] = False
        out[f"{actor}_direction"] = 0.0

    for sector, actor_directions in TEAM_STATIC_FLOW_DIRECTIONS.items():
        sector_mask = out["gics_sector"].astype(str).eq(sector)
        for actor, direction in actor_directions.items():
            if actor not in actor_cols:
                continue
            out.loc[sector_mask, f"{actor}_selected"] = True
            out.loc[sector_mask, f"{actor}_direction"] = float(direction)
    return out


def compute_flow_score(df: pd.DataFrame, actor_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    contributions = []
    selected_flags = []
    for actor in actor_cols:
        z = out[f"{actor}_flow_z"].fillna(0.0)
        direction = out[f"{actor}_direction"].fillna(0.0)
        selected = out[f"{actor}_selected"].fillna(False).astype(bool)
        selected_flags.append(selected.astype(int))
        contributions.append((direction * z).where(selected, 0.0))
    contribution_df = pd.concat(contributions, axis=1) if contributions else pd.DataFrame(index=out.index)
    selected_df = pd.concat(selected_flags, axis=1) if selected_flags else pd.DataFrame(index=out.index)
    out["flow_selected_actor_count"] = selected_df.sum(axis=1).astype(int)
    selected_count = out["flow_selected_actor_count"].replace(0, np.nan)
    out["flow_score_raw"] = contribution_df.sum(axis=1) / selected_count
    out["flow_score_raw"] = out["flow_score_raw"].fillna(0.0)
    selected_names = []
    for idx in out.index:
        names = [actor for actor in actor_cols if bool(out.at[idx, f"{actor}_selected"])]
        selected_names.append("|".join(names))
    out["flow_selected_actors"] = selected_names
    return out


def make_flow_rank(df: pd.DataFrame, lookback: int, min_obs: int, actor_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    obs_cols = [f"{actor}_predictive_obs" for actor in actor_cols]
    out["flow_obs"] = out[obs_cols].max(axis=1).fillna(0).astype(int)
    tradable = (
        (out["open"] > 0)
        & (out["close"] > 0)
        & (out["volume"] > 0)
        & (out["amount"] > 0)
    )
    out["flow_valid"] = tradable & (out["flow_obs"] >= min_obs) & np.isfinite(out["flow_score_raw"])
    out["flow_rank"] = np.nan
    out["flow_score"] = np.nan
    valid = out["flow_valid"]
    out.loc[valid, "flow_rank"] = out.loc[valid].groupby("date")[
        "flow_score_raw"
    ].rank(method="first", ascending=False)
    n_by_date = out.loc[valid].groupby("date")["ticker"].transform("count")
    ranks = out.loc[valid, "flow_rank"]
    scores = np.where(n_by_date > 1, 1.0 - (ranks - 1.0) / (n_by_date - 1.0), 1.0)
    out.loc[valid, "flow_score"] = scores
    out["flow_lookback"] = lookback
    return out


def validate_no_lookahead_flow(df: pd.DataFrame) -> None:
    forbidden = {"next_oc_ret", "next_open", "next_close", "future_return", "next_return", "label"}
    leaked = sorted(forbidden & set(df.columns))
    if leaked:
        raise ValueError(f"Forbidden look-ahead columns present before save: {leaked}")


def output_columns(actor_cols: list[str]) -> list[str]:
    cols = [
        "date",
        "ticker",
        "name",
        "gics_sector",
        "flow_score_raw",
        "flow_score",
        "flow_rank",
        "flow_valid",
        "flow_lookback",
        "flow_obs",
        "flow_selected_actor_count",
        "flow_selected_actors",
    ]
    for actor in actor_cols:
        cols.extend(
            [
                f"{actor}_flow_ratio",
                f"{actor}_flow_z",
                f"{actor}_predictive_corr",
                f"{actor}_predictive_corr_shrunk",
                f"{actor}_predictive_obs",
                f"{actor}_selected",
                f"{actor}_direction",
            ]
        )
    return cols


def save_flow_rank(
    df: pd.DataFrame,
    out_path: str | Path,
    actor_cols: list[str],
    corr_threshold: float,
    actor_selection_mode: str,
) -> None:
    out = df[output_columns(actor_cols)].copy()
    validate_no_lookahead_flow(out)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(out_path, index=False)
    no_actor = out[(out["flow_valid"]) & (out["flow_selected_actor_count"] == 0)][
        ["date", "ticker", "name", "gics_sector", "flow_obs"]
    ].copy()
    report_path = Path(out_path).with_name(Path(out_path).stem + "_no_selected_actor_report.csv")
    no_actor.to_csv(report_path, index=False)
    meta_path = Path(out_path).with_name(Path(out_path).stem + "_metadata.json")
    meta = {
        "score_formula": "mean(actor_direction * actor_flow_z) over selected actors",
        "corr_threshold": corr_threshold,
        "actor_selection_mode": actor_selection_mode,
        "team_static_includes_weak_directional_classes": actor_selection_mode == "team_static",
        "team_static_flow_directions": TEAM_STATIC_FLOW_DIRECTIONS if actor_selection_mode == "team_static" else None,
        "no_selected_actor_rows": int(len(no_actor)),
        "no_selected_actor_report": str(report_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def build_flow_rank(
    price_path: str | Path,
    flow_path: str | Path,
    output_path: str | Path,
    lookback: int = 20,
    min_obs: int = 20,
    shrink_k: float = 20.0,
    corr_threshold: float = 0.10,
    actor_selection_mode: str | None = None,
) -> pd.DataFrame:
    mode = actor_selection_mode or os.getenv("FLOW_ACTOR_SELECTION_MODE", "dynamic")
    if mode not in ACTOR_SELECTION_MODES:
        raise ValueError(f"actor_selection_mode must be one of {sorted(ACTOR_SELECTION_MODES)}, got {mode}")
    _, _, df = load_price_and_flow(price_path, flow_path)
    df = compute_internal_next_open_to_close_label(df)
    df = normalize_actor_flows(df, FLOW_ACTOR_COLUMNS)
    sanity_check_flow_units(df, FLOW_ACTOR_COLUMNS)
    df = compute_rolling_flow_zscore(df, FLOW_ACTOR_COLUMNS, lookback=lookback, min_obs=min_obs)
    df = compute_predictive_actor_corr(df, FLOW_ACTOR_COLUMNS, lookback=lookback, min_obs=min_obs)
    df = shrink_predictive_corr(df, FLOW_ACTOR_COLUMNS, shrink_k=shrink_k)
    if mode == "team_static":
        df = select_team_static_actors(df, FLOW_ACTOR_COLUMNS)
    else:
        df = select_predictive_actors(df, FLOW_ACTOR_COLUMNS, corr_threshold=corr_threshold)
    df = compute_flow_score(df, FLOW_ACTOR_COLUMNS)
    df = make_flow_rank(df, lookback=lookback, min_obs=min_obs, actor_cols=FLOW_ACTOR_COLUMNS)
    save_flow_rank(df, output_path, FLOW_ACTOR_COLUMNS, corr_threshold=corr_threshold, actor_selection_mode=mode)
    return df


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create flow/price rank CSV.")
    parser.add_argument("--price-input", default=base / "sector_all_merged.csv")
    parser.add_argument("--flow-input", default=base / "sector_fund_flow.csv")
    parser.add_argument("--output", default=base / "flow_rank.csv")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--min-obs", type=int, default=20)
    parser.add_argument("--shrink-k", type=float, default=20.0)
    parser.add_argument("--corr-threshold", type=float, default=0.10)
    parser.add_argument("--actor-selection-mode", choices=sorted(ACTOR_SELECTION_MODES), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_flow_rank(
        price_path=args.price_input,
        flow_path=args.flow_input,
        output_path=args.output,
        lookback=args.lookback,
        min_obs=args.min_obs,
        shrink_k=args.shrink_k,
        corr_threshold=args.corr_threshold,
        actor_selection_mode=args.actor_selection_mode,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

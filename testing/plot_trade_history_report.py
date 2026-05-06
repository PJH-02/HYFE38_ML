from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TRADE_CSV_PATHS: list[str] = [
    # Example:
    # r"C:\Users\박제형\Desktop\ML6\pilot\out_20260401_zai_flash_daily\A0\trade_history.csv",
    # r"C:\Users\박제형\Desktop\ML6\pilot\out_20260401_zai_flash_daily\A5\trade_history.csv",
]
OUTPUT_PNG = Path(__file__).resolve().parent / "trade_history_report.png"

REQUIRED_COLUMNS = {
    "strategy",
    "execution_date",
    "ticker",
    "nav_open_pre",
    "nav_close",
    "daily_return",
    "trade_turnover_value",
    "day_turnover_value",
    "day_commission",
}

BENCHMARK_PRICE_COLUMNS = ("index_close", "kospi_close", "close")


def infer_label(path: Path, strategy: str) -> str:
    parts = {part.lower() for part in path.parts}
    prefix = "pilot" if "pilot" in parts else ("testing" if "testing" in parts else path.parent.parent.name)
    parent = path.parent.name
    if parent and parent.lower() != strategy.lower():
        return f"{prefix} {parent}"
    return f"{prefix} {strategy}"


def load_trade_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    df["execution_date"] = pd.to_datetime(df["execution_date"])
    df["strategy"] = df["strategy"].astype(str)
    strategy = str(df["strategy"].iloc[0]) if not df.empty else csv_path.parent.name
    df["series_label"] = infer_label(csv_path, strategy)
    for col in ["nav_open_pre", "nav_close", "daily_return", "trade_turnover_value", "day_turnover_value", "day_commission"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def make_daily_panel(trades: pd.DataFrame) -> pd.DataFrame:
    daily = (
        trades.sort_values(["series_label", "execution_date", "ticker"])
        .groupby(["series_label", "execution_date"], as_index=False)
        .agg(
            nav_open_pre=("nav_open_pre", "first"),
            nav_close=("nav_close", "first"),
            daily_return=("daily_return", "first"),
            day_turnover_value=("day_turnover_value", "first"),
            day_commission=("day_commission", "first"),
            summed_trade_turnover=("trade_turnover_value", "sum"),
        )
    )
    daily["turnover_ratio"] = daily["day_turnover_value"] / daily["nav_open_pre"].where(daily["nav_open_pre"] > 0)
    return daily


def infer_initial_nav(group: pd.DataFrame) -> float:
    group = group.sort_values("execution_date")
    if group.empty:
        return 0.0
    first_return = float(group["daily_return"].iloc[0])
    first_nav = float(group["nav_close"].iloc[0])
    if first_return > -1.0:
        return first_nav / (1.0 + first_return)
    return float(group["nav_open_pre"].iloc[0])


def load_benchmark_prices(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "date" not in df.columns:
        raise ValueError(f"{csv_path} missing required benchmark date column: date")

    price_col = next((col for col in BENCHMARK_PRICE_COLUMNS if col in df.columns), None)
    if price_col is None:
        raise ValueError(f"{csv_path} missing one of benchmark price columns: {list(BENCHMARK_PRICE_COLUMNS)}")

    prices = df[["date", price_col]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices[price_col] = pd.to_numeric(prices[price_col], errors="coerce")
    prices = prices.dropna(subset=["date", price_col])
    prices = prices[prices[price_col] > 0]

    grouped = prices.groupby("date")[price_col].agg(["first", "min", "max"]).reset_index()
    inconsistent = grouped[(grouped["max"] - grouped["min"]).abs() > 1e-9]
    if not inconsistent.empty:
        sample = inconsistent["date"].dt.strftime("%Y-%m-%d").head(5).tolist()
        raise ValueError(f"{csv_path} has inconsistent benchmark prices on the same date: {sample}")

    return grouped.rename(columns={"first": "index_close"})[["date", "index_close"]].sort_values("date")


def make_benchmark_daily(
    price_path: str | Path,
    execution_dates: pd.Series,
    initial_nav: float,
    label: str = "BM",
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(execution_dates).drop_duplicates()).sort_values()
    if dates.empty:
        raise ValueError("Cannot build benchmark without execution dates.")
    if initial_nav <= 0:
        raise ValueError("Cannot build benchmark with non-positive initial NAV.")

    prices = load_benchmark_prices(price_path)
    first_date = dates.iloc[0]
    prev_prices = prices[prices["date"] < first_date]
    if prev_prices.empty:
        first_price = prices.loc[prices["date"] == first_date, "index_close"]
        if first_price.empty:
            raise ValueError(f"Benchmark has no price for first execution date {first_date.date()}.")
        synthetic_prev = float(first_price.iloc[0])
    else:
        synthetic_prev = float(prev_prices.iloc[-1]["index_close"])

    date_frame = pd.DataFrame({"execution_date": dates})
    aligned = date_frame.merge(prices.rename(columns={"date": "execution_date"}), on="execution_date", how="left")
    if aligned["index_close"].isna().any():
        missing = aligned.loc[aligned["index_close"].isna(), "execution_date"].dt.strftime("%Y-%m-%d").head(10).tolist()
        raise ValueError(f"Benchmark missing index close for execution dates: {missing}")

    previous_close = aligned["index_close"].shift(1)
    previous_close.iloc[0] = synthetic_prev
    returns = aligned["index_close"] / previous_close - 1.0
    nav_close = initial_nav * (1.0 + returns).cumprod()
    nav_open_pre = nav_close.shift(1)
    nav_open_pre.iloc[0] = initial_nav

    return pd.DataFrame(
        {
            "series_label": label,
            "execution_date": aligned["execution_date"],
            "nav_open_pre": nav_open_pre.astype(float),
            "nav_close": nav_close.astype(float),
            "daily_return": returns.astype(float),
            "day_turnover_value": 0.0,
            "day_commission": 0.0,
            "summed_trade_turnover": 0.0,
            "turnover_ratio": 0.0,
        }
    )


def compute_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in daily.groupby("series_label", sort=False):
        group = group.sort_values("execution_date")
        returns = group["daily_return"].astype(float)
        initial_nav = infer_initial_nav(group)
        final_nav = float(group["nav_close"].iloc[-1])
        drawdown = group["nav_close"] / group["nav_close"].cummax() - 1.0
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
        annual_vol = float(returns.std(ddof=0) * math.sqrt(252)) if len(returns) else 0.0
        sharpe = float((returns.mean() * 252) / annual_vol) if annual_vol > 0 else 0.0
        total_commission = float(group["day_commission"].sum())
        rows.append(
            {
                "series_label": label,
                "initial_nav": initial_nav,
                "final_nav": final_nav,
                "total_return": final_nav / initial_nav - 1.0 if initial_nav > 0 else 0.0,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "turnover_multiple": float(group["day_turnover_value"].sum() / initial_nav) if initial_nav > 0 else 0.0,
                "avg_turnover_ratio": float(group["turnover_ratio"].mean()),
                "total_commission": total_commission,
                "commission_drag": total_commission / initial_nav if initial_nav > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_report(daily: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> None:
    labels = metrics["series_label"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(22, 10), constrained_layout=True)
    ax_nav, ax_return, ax_sharpe, ax_mdd, ax_turnover, ax_commission = axes.flatten()

    for label, group in daily.groupby("series_label", sort=False):
        group = group.sort_values("execution_date")
        ax_nav.plot(group["execution_date"], group["nav_close"], marker="o", linewidth=1.8, label=label)
    ax_nav.set_title("NAV / PnL Path")
    ax_nav.set_xlabel("Execution Date")
    ax_nav.set_ylabel("NAV")
    ax_nav.grid(True, alpha=0.3)
    ax_nav.legend()

    ax_return.bar(labels, metrics["total_return"] * 100.0)
    ax_return.set_title("Total Return")
    ax_return.set_ylabel("%")
    ax_return.tick_params(axis="x", rotation=30)
    ax_return.grid(True, axis="y", alpha=0.3)

    ax_sharpe.bar(labels, metrics["sharpe"])
    ax_sharpe.set_title("Sharpe")
    ax_sharpe.tick_params(axis="x", rotation=30)
    ax_sharpe.grid(True, axis="y", alpha=0.3)

    ax_mdd.bar(labels, metrics["max_drawdown"] * 100.0)
    ax_mdd.set_title("Max Drawdown")
    ax_mdd.set_ylabel("%")
    ax_mdd.tick_params(axis="x", rotation=30)
    ax_mdd.grid(True, axis="y", alpha=0.3)

    ax_turnover.bar(labels, metrics["turnover_multiple"])
    ax_turnover.set_title("Total Turnover / Initial NAV")
    ax_turnover.tick_params(axis="x", rotation=30)
    ax_turnover.grid(True, axis="y", alpha=0.3)

    ax_commission.bar(labels, metrics["commission_drag"] * 100.0)
    ax_commission.set_title("Commission Drag")
    ax_commission.set_ylabel("% of initial NAV")
    ax_commission.tick_params(axis="x", rotation=30)
    ax_commission.grid(True, axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NAV, return, Sharpe, and turnover from trade_history.csv files.")
    parser.add_argument("csv_paths", nargs="*", type=Path, help="One or more trade_history.csv paths.")
    parser.add_argument("--out", type=Path, default=OUTPUT_PNG)
    parser.add_argument("--benchmark-price-path", type=Path, help="CSV containing date and index_close/kospi_close for BM buy-and-hold.")
    parser.add_argument("--benchmark-label", default="BM")
    parser.add_argument("--metrics-out", type=Path, help="Optional CSV path for calculated metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.csv_paths or [Path(path) for path in TRADE_CSV_PATHS]
    if not paths:
        raise SystemExit("No trade CSV paths provided. Set TRADE_CSV_PATHS or pass paths on the command line.")
    trades = pd.concat([load_trade_csv(path) for path in paths], ignore_index=True)
    daily = make_daily_panel(trades)
    if args.benchmark_price_path:
        first_label = daily["series_label"].iloc[0]
        initial_nav = infer_initial_nav(daily[daily["series_label"] == first_label])
        benchmark = make_benchmark_daily(
            args.benchmark_price_path,
            daily["execution_date"],
            initial_nav,
            label=args.benchmark_label,
        )
        daily = pd.concat([daily, benchmark], ignore_index=True)
    metrics = compute_metrics(daily)
    plot_report(daily, metrics, args.out)
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.metrics_out, index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))
    print(f"saved {args.out}")
    if args.metrics_out:
        print(f"saved {args.metrics_out}")


if __name__ == "__main__":
    main()

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


def compute_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in daily.groupby("series_label", sort=False):
        group = group.sort_values("execution_date")
        returns = group["daily_return"].astype(float)
        first_return = float(returns.iloc[0]) if len(returns) else 0.0
        first_nav = float(group["nav_close"].iloc[0]) if len(group) else 0.0
        initial_nav = first_nav / (1.0 + first_return) if first_return > -1.0 else float(group["nav_open_pre"].iloc[0])
        final_nav = float(group["nav_close"].iloc[-1])
        annual_vol = float(returns.std(ddof=0) * math.sqrt(252)) if len(returns) else 0.0
        sharpe = float((returns.mean() * 252) / annual_vol) if annual_vol > 0 else 0.0
        rows.append(
            {
                "series_label": label,
                "initial_nav": initial_nav,
                "final_nav": final_nav,
                "total_return": final_nav / initial_nav - 1.0 if initial_nav > 0 else 0.0,
                "sharpe": sharpe,
                "turnover_multiple": float(group["day_turnover_value"].sum() / initial_nav) if initial_nav > 0 else 0.0,
                "avg_turnover_ratio": float(group["turnover_ratio"].mean()),
                "total_commission": float(group["day_commission"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_report(daily: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> None:
    labels = metrics["series_label"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    ax_nav, ax_return, ax_sharpe, ax_turnover = axes.flatten()

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

    ax_turnover.bar(labels, metrics["turnover_multiple"])
    ax_turnover.set_title("Total Turnover / Initial NAV")
    ax_turnover.tick_params(axis="x", rotation=30)
    ax_turnover.grid(True, axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NAV, return, Sharpe, and turnover from trade_history.csv files.")
    parser.add_argument("csv_paths", nargs="*", type=Path, help="One or more trade_history.csv paths.")
    parser.add_argument("--out", type=Path, default=OUTPUT_PNG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.csv_paths or [Path(path) for path in TRADE_CSV_PATHS]
    if not paths:
        raise SystemExit("No trade CSV paths provided. Set TRADE_CSV_PATHS or pass paths on the command line.")
    trades = pd.concat([load_trade_csv(path) for path in paths], ignore_index=True)
    daily = make_daily_panel(trades)
    metrics = compute_metrics(daily)
    plot_report(daily, metrics, args.out)
    print(metrics.to_string(index=False))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

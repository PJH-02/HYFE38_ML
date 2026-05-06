from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_DIRS = {
    "A0": "A0",
    "A1": "A1_k10",
    "A2b": "A2b_k10",
    "A3": "A3_k10",
    "A4": "A4_k10",
    "A5": "A5_k10",
}
BENCHMARK_LABEL = "KOSPI_BUY_HOLD"


def infer_initial_nav(group: pd.DataFrame) -> float:
    ordered = group.sort_values("execution_date")
    if ordered.empty:
        return 0.0
    first_return = float(ordered["daily_return"].iloc[0])
    first_nav = float(ordered["nav_close"].iloc[0])
    if first_return > -1.0:
        return first_nav / (1.0 + first_return)
    return float(ordered["nav_open_pre"].iloc[0])


def load_model_daily(out_root: Path, label: str, directory: str) -> pd.DataFrame:
    trade_path = out_root / directory / "trade_history.csv"
    if not trade_path.exists():
        raise FileNotFoundError(f"Missing trade history for {label}: {trade_path}")
    trades = pd.read_csv(trade_path, encoding="utf-8-sig")
    required = {
        "execution_date",
        "ticker",
        "nav_open_pre",
        "nav_close",
        "daily_return",
        "day_turnover_value",
        "day_commission",
        "trade_turnover_value",
    }
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"{trade_path} missing columns: {sorted(missing)}")
    if trades.empty:
        raise ValueError(f"{trade_path} is empty")
    trades["execution_date"] = pd.to_datetime(trades["execution_date"])
    for col in [
        "nav_open_pre",
        "nav_close",
        "daily_return",
        "day_turnover_value",
        "day_commission",
        "trade_turnover_value",
    ]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce").fillna(0.0)
    daily = (
        trades.sort_values(["execution_date", "ticker"])
        .groupby("execution_date", as_index=False)
        .agg(
            nav_open_pre=("nav_open_pre", "first"),
            nav_close=("nav_close", "first"),
            daily_return=("daily_return", "first"),
            turnover_value=("day_turnover_value", "first"),
            commission=("day_commission", "first"),
            summed_trade_turnover=("trade_turnover_value", "sum"),
        )
    )
    daily["model"] = label
    daily["turnover_ratio"] = daily["turnover_value"] / daily["nav_open_pre"].where(daily["nav_open_pre"] > 0)
    return daily


def load_kospi_buy_hold(rank_panel: Path, execution_dates: pd.Series, initial_nav: float) -> pd.DataFrame:
    panel = pd.read_csv(rank_panel, usecols=["date", "index_close"], parse_dates=["date"])
    prices = (
        panel.dropna(subset=["date", "index_close"])
        .assign(index_close=lambda df: pd.to_numeric(df["index_close"], errors="coerce"))
        .dropna(subset=["index_close"])
        .groupby("date", as_index=False)["index_close"]
        .first()
        .sort_values("date")
    )
    dates = pd.Series(pd.to_datetime(execution_dates).drop_duplicates()).sort_values()
    aligned = pd.DataFrame({"execution_date": dates}).merge(prices.rename(columns={"date": "execution_date"}), on="execution_date", how="left")
    if aligned["index_close"].isna().any():
        missing = aligned.loc[aligned["index_close"].isna(), "execution_date"].dt.strftime("%Y-%m-%d").head(10).tolist()
        raise ValueError(f"KOSPI benchmark missing execution dates: {missing}")
    first_date = dates.iloc[0]
    previous = prices[prices["date"] < first_date]
    previous_close = float(previous.iloc[-1]["index_close"]) if not previous.empty else float(aligned.iloc[0]["index_close"])
    shifted = aligned["index_close"].shift(1)
    shifted.iloc[0] = previous_close
    returns = aligned["index_close"] / shifted - 1.0
    nav_close = initial_nav * (1.0 + returns).cumprod()
    nav_open_pre = nav_close.shift(1)
    nav_open_pre.iloc[0] = initial_nav
    return pd.DataFrame(
        {
            "execution_date": aligned["execution_date"],
            "nav_open_pre": nav_open_pre.astype(float),
            "nav_close": nav_close.astype(float),
            "daily_return": returns.astype(float),
            "turnover_value": 0.0,
            "commission": 0.0,
            "summed_trade_turnover": 0.0,
            "model": BENCHMARK_LABEL,
            "turnover_ratio": 0.0,
        }
    )


def compute_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in daily.groupby("model", sort=False, observed=False):
        group = group.sort_values("execution_date")
        returns = group["daily_return"].astype(float)
        initial_nav = infer_initial_nav(group)
        final_nav = float(group["nav_close"].iloc[-1])
        total_return = final_nav / initial_nav - 1.0 if initial_nav > 0 else 0.0
        annual_vol = float(returns.std(ddof=0) * math.sqrt(252)) if len(returns) else 0.0
        sharpe = float((returns.mean() * 252) / annual_vol) if annual_vol > 0 else 0.0
        drawdown = group["nav_close"] / group["nav_close"].cummax() - 1.0
        rows.append(
            {
                "model": model,
                "start_date": group["execution_date"].iloc[0].date().isoformat(),
                "end_date": group["execution_date"].iloc[-1].date().isoformat(),
                "initial_nav": initial_nav,
                "final_nav": final_nav,
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
                "turnover": float(group["turnover_value"].sum() / initial_nav) if initial_nav > 0 else 0.0,
                "avg_turnover_ratio": float(group["turnover_ratio"].mean()),
                "total_commission": float(group["commission"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_pnl(daily: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    for model, group in daily.groupby("model", sort=False, observed=False):
        group = group.sort_values("execution_date")
        ax.plot(group["execution_date"], group["nav_close"], linewidth=1.8, label=model)
    ax.set_title("Full-period NAV / PnL")
    ax.set_xlabel("Execution date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame, output: Path) -> None:
    labels = metrics["model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, column, title, pct in [
        (axes[0], "total_return", "Total Return", True),
        (axes[1], "sharpe", "Sharpe", False),
        (axes[2], "max_drawdown", "Max Drawdown", True),
    ]:
        values = metrics[column] * 100.0 if pct else metrics[column]
        ax.bar(labels, values)
        ax.set_title(title)
        if pct:
            ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final full-period model and benchmark report.")
    parser.add_argument("--out-root", type=Path, default=Path("backtest_state/out_actions_flow_v2_a0_a5_k10"))
    parser.add_argument("--rank-panel", type=Path, default=Path("rank_panel.csv"))
    parser.add_argument("--report-dir", type=Path, default=Path("backtest_state/final_report"))
    args = parser.parse_args()

    daily_parts = [load_model_daily(args.out_root, label, directory) for label, directory in MODEL_DIRS.items()]
    execution_dates = daily_parts[0]["execution_date"]
    initial_nav = infer_initial_nav(daily_parts[0])
    daily_parts.append(load_kospi_buy_hold(args.rank_panel, execution_dates, initial_nav))
    daily = pd.concat(daily_parts, ignore_index=True)
    order = list(MODEL_DIRS) + [BENCHMARK_LABEL]
    daily["model"] = pd.Categorical(daily["model"], categories=order, ordered=True)
    daily = daily.sort_values(["model", "execution_date"]).reset_index(drop=True)
    metrics = compute_metrics(daily)

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(report_dir / "daily_pnl_panel.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(report_dir / "model_benchmark_metrics.csv", index=False, encoding="utf-8-sig")
    plot_pnl(daily, report_dir / "pnl_all_models_vs_benchmark.png")
    plot_metric_bars(metrics, report_dir / "return_sharpe_mdd_bars.png")
    print(metrics.to_string(index=False))
    print(f"saved {report_dir}")


if __name__ == "__main__":
    main()

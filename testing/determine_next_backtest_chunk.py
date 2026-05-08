from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def month_ranges(rank_panel: Path, start_date: str | None = None, end_date: str | None = None) -> list[str]:
    panel = pd.read_csv(rank_panel, usecols=["date"], parse_dates=["date"])
    dates = sorted(panel["date"].dropna().drop_duplicates())
    decision_dates = dates[:-1]
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None
    if start_ts is not None:
        decision_dates = [date for date in decision_dates if pd.Timestamp(date) >= start_ts]
    if end_ts is not None:
        decision_dates = [date for date in decision_dates if pd.Timestamp(date) <= end_ts]
    months = sorted({pd.Timestamp(date).strftime("%Y-%m") for date in decision_dates})
    return months


def next_chunk(months: list[str], state_path: Path, months_per_run: int) -> tuple[str | None, str | None, bool]:
    if not months:
        return None, None, False
    last_completed: str | None = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        last_completed = state.get("last_completed_end_month")
    start_index = 0
    if last_completed in months:
        start_index = months.index(last_completed) + 1
    if start_index >= len(months):
        return None, None, False
    end_index = min(start_index + months_per_run - 1, len(months) - 1)
    return months[start_index], months[end_index], True


def main() -> None:
    parser = argparse.ArgumentParser(description="Determine the next committed 6-month LLM backtest chunk.")
    parser.add_argument("--rank-panel", type=Path, default=Path("rank_panel.csv"))
    parser.add_argument("--state-path", type=Path, default=Path("backtest_state/chunk_state.json"))
    parser.add_argument("--months-per-run", type=int, default=6)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    months = month_ranges(args.rank_panel, args.start_date, args.end_date)
    start_month, end_month, should_run = next_chunk(months, args.state_path, args.months_per_run)
    print(f"should_run={'1' if should_run else '0'}")
    print(f"start_month={start_month or ''}")
    print(f"end_month={end_month or ''}")
    print(f"first_month={months[0] if months else ''}")
    print(f"last_month={months[-1] if months else ''}")


if __name__ == "__main__":
    main()

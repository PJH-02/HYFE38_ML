from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

LLM_STRATEGIES = {
    "A2a": {
        "script": "A2a_llm_opaque_rank_allocator.py",
        "out_dir": "A2a_k{top_k}",
    },
    "A2b": {
        "script": "A2b_llm_semantic_rank_allocator.py",
        "out_dir": "A2b_k{top_k}",
    },
    "A3": {
        "script": "A3_llm_policy_pack_allocator.py",
        "out_dir": "A3_k{top_k}",
    },
    "A4": {
        "script": "A4_rule_based_llm_blend.py",
        "out_dir": "A4_k{top_k}",
    },
}


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"{stamp} {message}", flush=True)


def month_ranges(
    rank_panel: Path,
    start_month: str | None = None,
    end_month: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[tuple[str, str, str, int]]:
    panel = pd.read_csv(rank_panel, usecols=["date"], parse_dates=["date"])
    dates = sorted(panel["date"].dropna().drop_duplicates())
    decision_dates = dates[:-1]
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None
    if start_ts is not None:
        decision_dates = [date for date in decision_dates if pd.Timestamp(date) >= start_ts]
    if end_ts is not None:
        decision_dates = [date for date in decision_dates if pd.Timestamp(date) <= end_ts]
    if not decision_dates:
        return []
    by_month: dict[str, list[pd.Timestamp]] = {}
    for date in decision_dates:
        ts = pd.Timestamp(date)
        by_month.setdefault(ts.strftime("%Y-%m"), []).append(ts)
    return [
        (
            month,
            values[0].date().isoformat(),
            values[-1].date().isoformat(),
            len(values),
        )
        for month, values in sorted(by_month.items())
        if (start_month is None or month >= start_month) and (end_month is None or month <= end_month)
    ]


def read_progress(out_root: Path, strategy: str, top_k: int) -> dict:
    out_dir_name = LLM_STRATEGIES[strategy]["out_dir"].format(top_k=top_k)
    out_dir = out_root / out_dir_name
    checkpoint_path = out_dir / "checkpoint_state.json"
    summary_path = out_dir / "summary.json"
    progress: dict = {"out_dir": str(out_dir)}
    if checkpoint_path.exists():
        progress["checkpoint"] = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if summary_path.exists():
        progress["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return progress


def strategy_out_dir(out_root: Path, strategy: str, top_k: int) -> Path:
    return out_root / LLM_STRATEGIES[strategy]["out_dir"].format(top_k=top_k)


def append_progress_log(out_root: Path, row: dict) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "monthly_progress.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    strategy = row.get("strategy")
    top_k = row.get("top_k")
    if strategy in LLM_STRATEGIES and top_k is not None:
        strategy_dir = strategy_out_dir(out_root, str(strategy), int(top_k))
        strategy_dir.mkdir(parents=True, exist_ok=True)
        strategy_path = strategy_dir / "monthly_progress.jsonl"
        with strategy_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def format_metric(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if percent:
        return f"{number:.2%}"
    return f"{number:.6g}"


def run_month(strategy: str, rank_panel: Path, out_root: Path, top_k: int, start_date: str, end_date: str, dry_run: bool) -> None:
    script = LLM_STRATEGIES[strategy]["script"]
    command = [
        PYTHON,
        "-u",
        script,
        "--rank-panel",
        str(rank_panel),
        "--top-k",
        str(top_k),
        "--out-root",
        str(out_root),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
    ]
    log(f"MONTH START strategy={strategy} top_k={top_k} start={start_date} end={end_date}")
    if dry_run:
        log("DRY RUN " + " ".join(command))
        return
    completed = subprocess.run(command, cwd=BASE_DIR)
    if completed.returncode != 0:
        raise SystemExit(f"{strategy} month {start_date}..{end_date} failed with exit code {completed.returncode}")
    progress = read_progress(out_root, strategy, top_k)
    completed_decisions = progress.get("checkpoint", {}).get("completed_decisions")
    summary = progress.get("summary", {})
    final_nav = summary.get("final_nav")
    total_return = summary.get("total_return")
    sharpe = summary.get("sharpe")
    max_drawdown = summary.get("max_drawdown")
    fallback_rate = summary.get("fallback_rate")
    log(
        f"MONTH DONE strategy={strategy} top_k={top_k} start={start_date} end={end_date} "
        f"completed_decisions={completed_decisions} "
        f"nav={format_metric(final_nav)} "
        f"return={format_metric(total_return, percent=True)} "
        f"sharpe={format_metric(sharpe)} "
        f"mdd={format_metric(max_drawdown, percent=True)} "
        f"fallback_rate={format_metric(fallback_rate, percent=True)}"
    )
    append_progress_log(
        out_root,
        {
            "strategy": strategy,
            "top_k": top_k,
            "start_date": start_date,
            "end_date": end_date,
            "completed_decisions": completed_decisions,
            "final_nav": final_nav,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "fallback_rate": fallback_rate,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one LLM strategy as monthly checkpointed chunks.")
    parser.add_argument("--strategy", choices=sorted(LLM_STRATEGIES), required=True)
    parser.add_argument("--rank-panel", type=Path, default=Path("rank_panel.csv"))
    parser.add_argument("--out-root", type=Path, default=Path("out_actions_flow_v2_a0_a5_k10"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--start-month", default=None, help="Optional inclusive YYYY-MM month start.")
    parser.add_argument("--end-month", default=None, help="Optional inclusive YYYY-MM month end.")
    parser.add_argument("--start-date", default=None, help="Optional inclusive decision-date start.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive decision-date end.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rank_panel = args.rank_panel.resolve()
    out_root = args.out_root.resolve()
    ranges = month_ranges(
        rank_panel,
        start_month=args.start_month,
        end_month=args.end_month,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if not ranges:
        raise SystemExit(f"No monthly decision ranges found in {rank_panel}")
    log(
        f"MONTHLY RUN strategy={args.strategy} top_k={args.top_k} "
        f"start_month={args.start_month or 'FIRST'} end_month={args.end_month or 'LAST'} "
        f"months={len(ranges)} out_root={out_root}"
    )
    append_progress_log(
        out_root,
        {
            "strategy": args.strategy,
            "top_k": args.top_k,
            "event": "monthly_run_start",
            "start_month": args.start_month,
            "end_month": args.end_month,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "months": len(ranges),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    for month, start_date, end_date, decision_count in ranges:
        log(f"MONTH QUEUED strategy={args.strategy} month={month} decisions={decision_count}")
        run_month(args.strategy, rank_panel, out_root, args.top_k, start_date, end_date, args.dry_run)
    append_progress_log(
        out_root,
        {
            "strategy": args.strategy,
            "top_k": args.top_k,
            "event": "monthly_run_complete",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    log(f"MONTHLY COMPLETE strategy={args.strategy} top_k={args.top_k}")


if __name__ == "__main__":
    main()

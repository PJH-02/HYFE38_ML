from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

CHILDREN: list[subprocess.Popen] = []
STOP_REQUESTED = False


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def top_k_values() -> list[int]:
    raw = os.getenv("TOP_K_VALUES", "10")
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("TOP_K_VALUES must contain at least one integer")
    return values


def out_root() -> Path:
    return Path(os.getenv("OUT_ROOT", str(BASE_DIR / "out_nvidia_k10"))).resolve()


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"{stamp} {message}", flush=True)


def terminate_children() -> None:
    for child in CHILDREN:
        if child.poll() is None:
            log(f"terminating child pid={child.pid}")
            child.terminate()
    for child in CHILDREN:
        if child.poll() is None:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log(f"killing child pid={child.pid}")
                child.kill()


def handle_stop(signum, frame) -> None:  # noqa: ANN001
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log(f"received signal={signum}; preserving checkpoints and stopping children")
    terminate_children()


def run_step(args: list[str], label: str) -> None:
    if STOP_REQUESTED:
        raise KeyboardInterrupt
    log(f"START {label}")
    completed = subprocess.run([PYTHON, *args], cwd=BASE_DIR)
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")
    log(f"DONE {label}")


def run_rank_build_if_needed() -> None:
    rank_panel = BASE_DIR / "rank_panel.csv"
    rebuild = env_flag("REBUILD_RANKS", False)
    if rank_panel.exists() and not rebuild:
        log("SKIP rank build; rank_panel.csv already exists")
        return
    run_step(["01_make_market_corr_rank.py"], "market rank")
    run_step(["02_make_flow_price_rank.py"], "flow rank")
    run_step(["03_make_rotation_rank.py"], "rotation rank")
    run_step(["04_merge_rank_panel.py"], "rank panel merge")


def run_fast_backtests(values: list[int], root: Path) -> None:
    if not env_flag("RUN_A0_A1_A5", True):
        log("SKIP A0/A1/A5")
        return
    top_k_text = ",".join(str(v) for v in values)
    run_step(["A0_equal_weight.py", "--out-root", str(root)], "A0")
    run_step(["A1_rule_based_rank_allocator.py", "--out-root", str(root), "--top-k", top_k_text], f"A1 top_k={top_k_text}")
    run_step(["A5_bayesian_winner_loser_allocator.py", "--out-root", str(root), "--top-k", top_k_text], f"A5 top_k={top_k_text}")


def spawn_llm(script: str, strategy: str, top_k: int, root: Path) -> subprocess.Popen:
    log_dir = root
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{strategy}_k{top_k}.run.log"
    stderr_path = log_dir / f"{strategy}_k{top_k}.err.log"
    stdout = stdout_path.open("a", encoding="utf-8")
    stderr = stderr_path.open("a", encoding="utf-8")
    child = subprocess.Popen(
        [
            PYTHON,
            "-u",
            script,
            "--rank-panel",
            "rank_panel.csv",
            "--top-k",
            str(top_k),
            "--out-root",
            str(root),
        ],
        cwd=BASE_DIR,
        stdout=stdout,
        stderr=stderr,
    )
    CHILDREN.append(child)
    log(f"SPAWN {strategy} top_k={top_k} pid={child.pid}")
    return child


def run_llm_backtests(values: list[int], root: Path) -> None:
    if not env_flag("RUN_LLM", True):
        log("SKIP LLM backtests")
        return
    scripts = [
        ("A2a_llm_opaque_rank_allocator.py", "A2a"),
        ("A2b_llm_semantic_rank_allocator.py", "A2b"),
        ("A3_llm_policy_pack_allocator.py", "A3"),
        ("A4_rule_based_llm_blend.py", "A4"),
    ]
    parallel = env_flag("RUN_LLM_PARALLEL", True)
    for top_k in values:
        if parallel:
            batch = [spawn_llm(script, name, top_k, root) for script, name in scripts]
            for child in batch:
                code = child.wait()
                if code != 0 and not STOP_REQUESTED:
                    raise SystemExit(f"LLM child pid={child.pid} failed with exit code {code}")
            log(f"DONE LLM batch top_k={top_k}")
        else:
            for script, name in scripts:
                child = spawn_llm(script, name, top_k, root)
                code = child.wait()
                if code != 0 and not STOP_REQUESTED:
                    raise SystemExit(f"{name} top_k={top_k} failed with exit code {code}")
                log(f"DONE {name} top_k={top_k}")
        if STOP_REQUESTED:
            raise KeyboardInterrupt


def run_a6(values: list[int], root: Path) -> None:
    if not env_flag("RUN_A6", True):
        log("SKIP A6")
        return
    mode = os.getenv("A6_MODE", "both")
    args = ["A6_bayesian_online_expert_allocator.py", "--rank-panel", "rank_panel.csv", "--out-root", str(root), "--mode", mode, "--top-k"]
    args.extend(str(v) for v in values)
    run_step(args, f"A6 mode={mode} top_k={','.join(str(v) for v in values)}")


def main() -> None:
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    values = top_k_values()
    root = out_root()
    root.mkdir(parents=True, exist_ok=True)
    log(f"RUN full backtest out_root={root} top_k={values}")
    try:
        run_rank_build_if_needed()
        run_fast_backtests(values, root)
        run_llm_backtests(values, root)
        run_a6(values, root)
    except KeyboardInterrupt:
        log("STOPPED; existing checkpoint/output files are preserved")
        raise SystemExit(130)
    log("COMPLETE full backtest")


if __name__ == "__main__":
    main()

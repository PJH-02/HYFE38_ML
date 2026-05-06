from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from A2a_llm_opaque_rank_allocator import (
    DecisionResult,
    REASON_CODE_ENUM,
    apply_top_k_filter,
    call_llm,
    create_id_mapping,
    generate_A1_base_weight,
    get_valid_universe,
    _is_placeholder_secret,
    load_rank_panel,
    load_llm_config,
    map_asset_id_to_ticker,
    normalize_sleeve_weights,
    parse_llm_json,
    repair_invalid_llm_output_once,
    run_llm_backtest,
    validate_llm_sleeve_weights,
    validate_top_k,
)
from A2b_llm_semantic_rank_allocator import build_semantic_packet


def load_policy_pack() -> dict[str, Any]:
    return {
        "constraints": {
            "rank_1_is_best": True,
            "higher_score_is_better": True,
            "sleeve_weight_sum": 1.0,
            "avoid_unnecessary_turnover": True,
            "avoid_excess_concentration": True,
        },
        "rank_conflict_rules": [
            "Strong agreement across ranks supports higher conviction.",
            "A single strong rank with two weak ranks should not dominate the sleeve.",
            "Two strong ranks with one weak rank supports moderate conviction.",
            "Weak ranks across all signals should receive zero or minimal weight.",
        ],
        "signal_definitions": {
            "market_rank": "regime-adjusted broad-market co-movement rank: risk-on favors higher recent rolling broad-market correlation; risk-off and neutral favor lower recent rolling broad-market correlation.",
            "flow_rank": "rank 1 is the strongest selected-actor investor-flow signal; selected actor z-scores are sign-adjusted by historical predictive correlation and equal-weighted.",
            "rotation_rank": "rank 1 is the strongest latest close-to-close cross-sectional price rotation.",
        },
        "reason_code_enum": sorted(REASON_CODE_ENUM),
    }


def build_policy_packet(
    day_df: pd.DataFrame,
    current_weights: dict[str, float],
    id_mapping: pd.DataFrame,
    decision_step: int = 0,
) -> dict[str, Any]:
    return build_semantic_packet(day_df, current_weights, id_mapping, decision_step)


def build_policy_prompt(packet: dict[str, Any], policy_pack: dict[str, Any], top_k: int = 10) -> str:
    prompt = {
        "instructions": [
            "Use only the provided ranks, scores, current weights, constraints, and policy pack.",
            "Interpret market_rank, flow_rank, and rotation_rank according to policy_pack.signal_definitions.",
            "Do not use outside knowledge, external labels, news, or memory.",
            "Allocate 100% of the predefined asset sleeve.",
            "Return JSON only with decision_step, target_weights, and portfolio_reason_codes.",
            "target_weights must be a list of objects: [{\"asset_id\":\"asset_001\",\"sleeve_weight\":0.25,\"reason_codes\":[\"rank_alignment_strong\"]}].",
        ],
        "constraints": {
            "max_nonzero_weights": min(top_k, len(packet["assets"])) if top_k < 10 else len(packet["assets"]),
            "sleeve_weight_sum": 1.0,
            "no_negative_weights": True,
        },
        "policy_pack": policy_pack,
        "packet": packet,
    }
    return json.dumps(prompt, ensure_ascii=True, separators=(",", ":"))


def parse_and_validate_reason_codes(parsed: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if not isinstance(parsed, dict):
        return False, ["response_not_json_object"]
    errors: list[str] = []
    portfolio_codes = parsed.get("portfolio_reason_codes", [])
    if portfolio_codes and (not isinstance(portfolio_codes, list) or set(portfolio_codes) - REASON_CODE_ENUM):
        errors.append("invalid_portfolio_reason_codes")
    for item in parsed.get("target_weights", []):
        if not isinstance(item, dict):
            continue
        codes = item.get("reason_codes", [])
        if codes and (not isinstance(codes, list) or set(codes) - REASON_CODE_ENUM):
            errors.append("invalid_reason_codes")
            break
    return not errors, errors


def generate_A3_weight(
    day_df: pd.DataFrame,
    current_weights: dict[str, float],
    top_k: int = 10,
    decision_step: int = 0,
    id_mapping: pd.DataFrame | None = None,
) -> DecisionResult:
    validate_top_k(top_k)
    id_mapping = id_mapping if id_mapping is not None else create_id_mapping(day_df)
    valid = get_valid_universe(day_df)
    base_weights = generate_A1_base_weight(day_df, top_k=top_k)
    if valid.empty:
        return DecisionResult({}, True, {"fallback_used": True, "fallback_reason": "empty_valid_universe"})
    packet = build_policy_packet(day_df, current_weights, id_mapping, decision_step)
    prompt = build_policy_prompt(packet, load_policy_pack(), top_k=top_k)
    response, call_meta = call_llm(prompt)
    parsed = parse_llm_json(response)
    valid_asset_ids = {asset["asset_id"] for asset in packet["assets"]}
    reason_ok, reason_errors = parse_and_validate_reason_codes(parsed)
    ok, asset_weights, errors = validate_llm_sleeve_weights(parsed, valid_asset_ids, top_k, require_reason_codes=True)
    errors = errors + ([] if reason_ok else reason_errors)
    repair_used = False
    if (not ok or not reason_ok) and parsed is not None:
        repaired, repair_used = repair_invalid_llm_output_once(parsed, valid_asset_ids, top_k)
        ok, asset_weights, errors = validate_llm_sleeve_weights(repaired, valid_asset_ids, top_k)
    if not ok:
        return DecisionResult(
            base_weights,
            True,
            {
                "fallback_used": True,
                "fallback_reason": "missing_api_key" if _is_placeholder_secret(str(load_llm_config().get("api_key", ""))) else "invalid_llm_output",
                "validation_errors": errors,
                "repair_used": repair_used,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                **call_meta,
            },
        )
    ticker_weights = map_asset_id_to_ticker(asset_weights, id_mapping)
    ticker_weights = apply_top_k_filter(normalize_sleeve_weights(ticker_weights, list(valid["ticker"].astype(str))), top_k)
    return DecisionResult(
        ticker_weights,
        False,
        {
            "fallback_used": False,
            "repair_used": repair_used,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            **call_meta,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A3 LLM policy-pack allocator.")
    parser.add_argument("--rank-panel", default="rank_panel.csv")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--out-root", default="out")
    args = parser.parse_args()
    panel = load_rank_panel(args.rank_panel)
    run_llm_backtest(panel, "A3_LLM_POLICY", args.top_k, Path(args.out_root) / f"A3_k{args.top_k}", generate_A3_weight)


if __name__ == "__main__":
    main()

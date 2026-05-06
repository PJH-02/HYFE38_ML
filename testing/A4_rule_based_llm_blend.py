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
    apply_top_k_filter,
    call_llm,
    classify_llm_failure,
    create_id_mapping,
    generate_A1_base_weight,
    get_valid_universe,
    load_rank_panel,
    map_asset_id_to_ticker,
    normalize_sleeve_weights,
    parse_llm_json,
    repair_invalid_llm_output_once,
    run_llm_backtest,
    validate_llm_sleeve_weights,
    validate_top_k,
)
from A3_llm_policy_pack_allocator import load_policy_pack
from A2b_llm_semantic_rank_allocator import build_semantic_packet


LAMBDA_VALUE = 0.5


def build_blend_llm_packet(
    day_df: pd.DataFrame,
    current_weights: dict[str, float],
    base_weights: dict[str, float],
    id_mapping: pd.DataFrame,
    decision_step: int = 0,
) -> dict[str, Any]:
    packet = build_semantic_packet(day_df, current_weights, id_mapping, decision_step)
    ticker_to_asset = dict(zip(id_mapping["ticker"].astype(str), id_mapping["asset_id"].astype(str)))
    base_by_asset = {ticker_to_asset[ticker]: weight for ticker, weight in base_weights.items() if ticker in ticker_to_asset}
    for asset in packet["assets"]:
        asset["base_sleeve_weight"] = float(base_by_asset.get(asset["asset_id"], 0.0))
    return packet


def build_blend_policy_prompt(packet: dict[str, Any], top_k: int = 10) -> str:
    prompt = {
        "instructions": [
            "Use only the provided ranks, scores, current weights, base weights, constraints, and policy pack.",
            "Interpret market_rank, flow_rank, and rotation_rank according to policy_pack.signal_definitions.",
            "Treat base_sleeve_weight as a rule-based reference, not as mandatory final weight.",
            "Do not use outside knowledge, external labels, news, or memory.",
            "Allocate 100% of the predefined asset sleeve.",
            "Return JSON only with target_weights and portfolio_reason_codes.",
            "target_weights must be a list of objects: [{\"asset_id\":\"asset_001\",\"sleeve_weight\":0.25,\"reason_codes\":[\"rank_alignment_strong\"]}].",
        ],
        "constraints": {
            "max_nonzero_weights": min(top_k, len(packet["assets"])) if top_k < 10 else len(packet["assets"]),
            "sleeve_weight_sum": 1.0,
            "no_negative_weights": True,
        },
        "policy_pack": load_policy_pack(),
        "packet": packet,
    }
    return json.dumps(prompt, ensure_ascii=True, separators=(",", ":"))


def generate_policy_llm_weight(
    day_df: pd.DataFrame,
    current_weights: dict[str, float],
    base_weights: dict[str, float],
    top_k: int = 10,
    decision_step: int = 0,
    id_mapping: pd.DataFrame | None = None,
) -> DecisionResult:
    id_mapping = id_mapping if id_mapping is not None else create_id_mapping(day_df)
    valid = get_valid_universe(day_df)
    packet = build_blend_llm_packet(day_df, current_weights, base_weights, id_mapping, decision_step)
    prompt = build_blend_policy_prompt(packet, top_k=top_k)
    response, call_meta = call_llm(prompt)
    parsed = parse_llm_json(response)
    valid_asset_ids = {asset["asset_id"] for asset in packet["assets"]}
    ok, asset_weights, errors = validate_llm_sleeve_weights(parsed, valid_asset_ids, top_k, require_reason_codes=True)
    repair_used = False
    if not ok and parsed is not None:
        repaired, repair_used = repair_invalid_llm_output_once(parsed, valid_asset_ids, top_k)
        ok, asset_weights, errors = validate_llm_sleeve_weights(repaired, valid_asset_ids, top_k)
    if not ok:
        return DecisionResult(
            base_weights,
            True,
            {
                "fallback_used": True,
                "fallback_reason": classify_llm_failure(call_meta, errors),
                "validation_errors": errors,
                "repair_used": repair_used,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                **call_meta,
            },
        )
    ticker_weights = map_asset_id_to_ticker(asset_weights, id_mapping)
    ticker_weights = normalize_sleeve_weights(ticker_weights, list(valid["ticker"].astype(str)))
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


def blend_weights(
    base_weights: dict[str, float],
    llm_weights: dict[str, float],
    lambda_value: float = LAMBDA_VALUE,
) -> dict[str, float]:
    tickers = sorted(set(base_weights) | set(llm_weights))
    raw = {
        ticker: lambda_value * base_weights.get(ticker, 0.0) + (1.0 - lambda_value) * llm_weights.get(ticker, 0.0)
        for ticker in tickers
    }
    return normalize_sleeve_weights(raw, tickers)


def generate_A4_weight(
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
    llm_decision = generate_policy_llm_weight(
        day_df,
        current_weights,
        base_weights,
        top_k=top_k,
        decision_step=decision_step,
        id_mapping=id_mapping,
    )
    if llm_decision.fallback_used:
        return DecisionResult(
            base_weights,
            True,
            {
                **llm_decision.log,
                "a4_blend_used": False,
                "lambda_value": LAMBDA_VALUE,
            },
        )
    blended = blend_weights(base_weights, llm_decision.weights, lambda_value=LAMBDA_VALUE)
    blended = apply_top_k_filter(blended, top_k)
    return DecisionResult(
        blended,
        False,
        {
            **llm_decision.log,
            "a4_blend_used": True,
            "lambda_value": LAMBDA_VALUE,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A4 rule-based / LLM blend allocator.")
    parser.add_argument("--rank-panel", default="rank_panel.csv")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--out-root", default="out")
    args = parser.parse_args()
    panel = load_rank_panel(args.rank_panel)
    run_llm_backtest(panel, "A4_RULE_LLM_BLEND", args.top_k, Path(args.out_root) / f"A4_k{args.top_k}", generate_A4_weight)


if __name__ == "__main__":
    main()

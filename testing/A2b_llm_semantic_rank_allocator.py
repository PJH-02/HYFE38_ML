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
    build_opaque_packet,
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


def build_semantic_packet(
    day_df: pd.DataFrame,
    current_weights: dict[str, float],
    id_mapping: pd.DataFrame,
    decision_step: int = 0,
) -> dict[str, Any]:
    opaque = build_opaque_packet(day_df, current_weights, id_mapping, decision_step)
    assets = []
    for item in opaque["assets"]:
        assets.append(
            {
                "asset_id": item["asset_id"],
                "market_rank": item["signal_1_rank"],
                "market_score": item["signal_1_score"],
                "flow_rank": item["signal_2_rank"],
                "flow_score": item["signal_2_score"],
                "rotation_rank": item["signal_3_rank"],
                "rotation_score": item["signal_3_score"],
                "current_sleeve_weight": item["current_sleeve_weight"],
            }
        )
    return {**opaque, "assets": assets}


def build_semantic_prompt(packet: dict[str, Any], top_k: int = 10) -> str:
    prompt = {
        "instructions": [
            "Use only the provided ranks and scores.",
            "market_rank: regime-adjusted broad-index co-movement rank. In risk-on regimes rank 1 favors higher recent rolling correlation to the broad index; in risk-off or neutral regimes rank 1 favors lower recent rolling correlation. It is a co-movement signal, not a standalone return forecast.",
            "market_score: 0 to 1 cross-sectional normalization of the regime-adjusted market_rank; higher is better under the active regime rule.",
            "flow_rank: rank 1 means the asset has the strongest investor-flow signal after selecting only flow actors whose past predictive correlation exceeded the threshold for that asset. Actor z-scores are direction-adjusted by the historical correlation sign and then equal-weighted.",
            "flow_score: 0 to 1 cross-sectional normalization of flow_rank; higher means more favorable historically predictive flow pressure.",
            "rotation_rank: rank 1 means the asset has the strongest latest completed weekly rotation signal among the valid universe. The signal combines equal-weighted components: 4-week broad-index-relative strength, valid leader persistence, and historical transition-pair frequency. Weekly signals are available only after the relevant week has completed and are then carried forward to daily decisions.",
            "rotation_score: 0 to 1 cross-sectional normalization of rotation_rank; higher means stronger completed-week relative-strength/leader/transition rotation evidence.",
            "All ranks are point-in-time signals available after the decision date close for next-open execution.",
            "Do not use outside knowledge, external labels, news, or memory.",
            "Allocate 100% of the predefined asset sleeve.",
            "Return JSON only with target_weights.",
            "target_weights must be a list of objects: [{\"asset_id\":\"asset_001\",\"sleeve_weight\":0.25}].",
        ],
        "constraints": {
            "max_nonzero_weights": min(top_k, len(packet["assets"])) if top_k < 10 else len(packet["assets"]),
            "sleeve_weight_sum": 1.0,
            "no_negative_weights": True,
        },
        "packet": packet,
    }
    return json.dumps(prompt, ensure_ascii=True, separators=(",", ":"))


def generate_A2b_weight(
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
    packet = build_semantic_packet(day_df, current_weights, id_mapping, decision_step)
    prompt = build_semantic_prompt(packet, top_k=top_k)
    response, call_meta = call_llm(prompt)
    parsed = parse_llm_json(response)
    valid_asset_ids = {asset["asset_id"] for asset in packet["assets"]}
    ok, asset_weights, errors = validate_llm_sleeve_weights(parsed, valid_asset_ids, top_k)
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
    parser = argparse.ArgumentParser(description="Run A2b LLM semantic rank allocator.")
    parser.add_argument("--rank-panel", default="rank_panel.csv")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--out-root", default="out")
    parser.add_argument("--start-date", default=None, help="Optional inclusive decision-date start.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive decision-date end.")
    args = parser.parse_args()
    panel = load_rank_panel(args.rank_panel)
    run_llm_backtest(
        panel,
        "A2b_LLM_SEMANTIC",
        args.top_k,
        Path(args.out_root) / f"A2b_k{args.top_k}",
        generate_A2b_weight,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()

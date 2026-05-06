from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from A0_equal_weight import (
        apply_regime_market_score,
        build_trade_rows,
        default_regime_path,
        load_regime_table,
        regime_info_for_date,
        regime_stock_ratio_for_date,
    )
except ModuleNotFoundError:
    from .A0_equal_weight import (
        apply_regime_market_score,
        build_trade_rows,
        default_regime_path,
        load_regime_table,
        regime_info_for_date,
        regime_stock_ratio_for_date,
    )


ALLOWED_TOP_K = {3, 5, 7, 10}
INITIAL_NAV = 1_000_000_000.0
COMMISSION_RATE = 0.00015
STOCK_RATIO = 1.0
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "glm-5.1"
DEFAULT_LLM_CONFIG_PATH = Path(__file__).resolve().parent / "zai_llm_config.json"
DAILY_RESULT_COLUMNS = [
    "decision_date",
    "execution_date",
    "strategy",
    "nav_close",
    "nav_open_pre",
    "nav_open_post",
    "overnight_pnl",
    "open_to_close_pnl",
    "commission",
    "turnover_value",
    "turnover_ratio",
    "daily_return",
    "stock_ratio",
    "fallback_used",
    "effective_num_positions",
]
WEIGHT_COLUMNS = [
    "decision_date",
    "execution_date",
    "ticker",
    "ETF_id",
    "sleeve_weight",
    "portfolio_weight",
]
TRADE_COLUMNS = [
    "strategy",
    "decision_date",
    "execution_date",
    "ticker",
    "ETF_id",
    "name",
    "open",
    "close",
    "prev_qty",
    "target_qty",
    "delta_qty",
    "side",
    "prev_value_open",
    "target_value_open",
    "delta_value_open",
    "trade_turnover_value",
    "day_turnover_value",
    "commission_allocated",
    "day_commission",
    "nav_open_pre",
    "nav_open_post",
    "nav_close",
    "daily_return",
    "stock_ratio",
    "cash_weight",
    "target_sleeve_weight",
    "target_portfolio_weight",
    "prev_portfolio_weight_open",
    "actual_portfolio_weight_open",
    "cash_after_rebalance",
    "rebalanced",
]


@dataclass
class DecisionResult:
    weights: dict[str, float]
    fallback_used: bool
    log: dict[str, Any]


def validate_top_k(top_k: int) -> None:
    if top_k not in ALLOWED_TOP_K:
        raise ValueError(f"top_k must be one of {sorted(ALLOWED_TOP_K)}, got {top_k}")


def load_rank_panel(path: str | Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    required = {
        "date",
        "ticker",
        "open",
        "close",
        "market_rank",
        "market_score",
        "flow_rank",
        "flow_score",
        "rotation_rank",
        "rotation_score",
        "tradable",
        "all_rank_valid",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"rank panel missing columns: {sorted(missing)}")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    if "ETF_id" not in panel.columns:
        panel = panel.merge(create_id_mapping(panel)[["ticker", "ETF_id"]], on="ticker", how="left")
    if "asset_id" not in panel.columns:
        panel = panel.merge(create_id_mapping(panel)[["ticker", "asset_id"]], on="ticker", how="left")
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def write_csv_records(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False)


def _is_placeholder_secret(value: str | None) -> bool:
    if not value:
        return True
    lowered = value.strip().lower()
    return lowered in {"", "put_zai_api_key_here", "replace_me", "your-api-key", "your-z.ai-api-key"} or "put_" in lowered


def _config_or_env(config: dict[str, Any], key: str, env_key: str, default: Any) -> Any:
    value = config.get(key)
    if value is not None:
        return value
    env_value = os.getenv(env_key)
    if env_value is not None and env_value != "":
        return env_value
    return default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _json_config_or_env(config: dict[str, Any], key: str, env_key: str) -> Any:
    if key in config:
        return config[key]
    raw = os.getenv(env_key)
    if raw is None or raw.strip() == "":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_key} must be valid JSON") from exc


@lru_cache(maxsize=1)
def load_llm_config() -> dict[str, Any]:
    config_path = Path(os.getenv("LLM_CONFIG_PATH", DEFAULT_LLM_CONFIG_PATH))
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    provider = str(config.get("provider") or os.getenv("LLM_PROVIDER") or ("zai" if config_path.exists() else "openai"))
    api_key = (
        config.get("api_key")
        or os.getenv("LLM_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
        or os.getenv("ZAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    base_url = str(
        config.get("base_url")
        or os.getenv("LLM_BASE_URL")
        or ("https://api.z.ai/api/paas/v4/" if provider == "zai" else "https://api.openai.com/v1/")
    )
    chat_url = str(config.get("chat_completions_url") or os.getenv("LLM_CHAT_COMPLETIONS_URL") or base_url.rstrip("/") + "/chat/completions")
    model = str(config.get("model") or os.getenv("LLM_MODEL") or DEFAULT_MODEL)
    timeout = int(_config_or_env(config, "timeout_seconds", "LLM_TIMEOUT_SECONDS", 60))
    temperature = float(_config_or_env(config, "temperature", "LLM_TEMPERATURE", 0.0))
    max_tokens = int(_config_or_env(config, "max_tokens", "LLM_MAX_TOKENS", 2048))
    max_retries = int(_config_or_env(config, "max_retries", "LLM_MAX_RETRIES", 1))
    retry_sleep_seconds = float(_config_or_env(config, "retry_sleep_seconds", "LLM_RETRY_SLEEP_SECONDS", 0.0))
    success_sleep_seconds = float(_config_or_env(config, "success_sleep_seconds", "LLM_SUCCESS_SLEEP_SECONDS", 0.0))
    response_format = _json_config_or_env(config, "response_format", "LLM_RESPONSE_FORMAT")
    if response_format is None and _env_flag("LLM_RESPONSE_FORMAT_JSON", False):
        response_format = {"type": "json_object"}
    thinking = _json_config_or_env(config, "thinking", "LLM_THINKING")
    chat_template_kwargs = _json_config_or_env(config, "chat_template_kwargs", "LLM_CHAT_TEMPLATE_KWARGS")
    return {
        **config,
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "chat_completions_url": chat_url,
        "model": model,
        "timeout_seconds": timeout,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
        "retry_sleep_seconds": retry_sleep_seconds,
        "success_sleep_seconds": success_sleep_seconds,
        "response_format": response_format,
        "thinking": thinking,
        "chat_template_kwargs": chat_template_kwargs,
    }


def create_id_mapping(panel: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(panel["ticker"].astype(str).dropna().unique())
    return pd.DataFrame(
        {
            "ticker": tickers,
            "ETF_id": [f"ETF_{i + 1:03d}" for i in range(len(tickers))],
            "asset_id": [f"asset_{i + 1:03d}" for i in range(len(tickers))],
        }
    )


def get_valid_universe(day_df: pd.DataFrame) -> pd.DataFrame:
    valid = day_df[
        (day_df["tradable"].astype(bool))
        & (day_df["all_rank_valid"].astype(bool))
        & day_df[["market_score", "flow_score", "rotation_score"]].notna().all(axis=1)
        & (day_df["open"] > 0)
        & (day_df["close"] > 0)
    ].copy()
    return valid.sort_values("ticker").reset_index(drop=True)


def normalize_sleeve_weights(raw_weights: dict[str, float], valid_tickers: list[str]) -> dict[str, float]:
    cleaned = {ticker: max(float(raw_weights.get(ticker, 0.0)), 0.0) for ticker in valid_tickers}
    total = sum(cleaned.values())
    if total <= 0.0:
        if not valid_tickers:
            return {}
        equal = 1.0 / len(valid_tickers)
        return {ticker: equal for ticker in valid_tickers}
    return {ticker: weight / total for ticker, weight in cleaned.items()}


def apply_top_k_filter(weights: dict[str, float], top_k: int) -> dict[str, float]:
    validate_top_k(top_k)
    if top_k == 10:
        return normalize_sleeve_weights(weights, list(weights))
    keep = {
        ticker
        for ticker, _ in sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        if weights[ticker] > 0.0
    }
    filtered = {ticker: (weight if ticker in keep else 0.0) for ticker, weight in weights.items()}
    return normalize_sleeve_weights(filtered, list(weights))


def compute_composite_score(day_df: pd.DataFrame) -> pd.Series:
    return (
        0.30 * day_df["market_score"].fillna(0.0)
        + 0.35 * day_df["flow_score"].fillna(0.0)
        + 0.35 * day_df["rotation_score"].fillna(0.0)
    )


def generate_A1_base_weight(day_df: pd.DataFrame, top_k: int = 10) -> dict[str, float]:
    try:
        from A1_rule_based_rank_allocator import generate_rule_based_weight  # type: ignore

        external = generate_rule_based_weight(day_df, top_k=top_k)
        if isinstance(external, tuple):
            external = external[0]
        if isinstance(external, pd.Series):
            return {str(ticker): float(weight) for ticker, weight in external.items()}
        if isinstance(external, dict):
            return {str(ticker): float(weight) for ticker, weight in external.items()}
    except Exception:
        pass
    valid = get_valid_universe(day_df)
    if valid.empty:
        return {}
    valid = valid.copy()
    valid["composite_score"] = compute_composite_score(valid)
    candidates = valid
    if top_k < 10:
        candidates = valid.sort_values(["composite_score", "ticker"], ascending=[False, True]).head(top_k)
    raw = {ticker: 0.0 for ticker in valid["ticker"].astype(str)}
    for row in candidates.itertuples(index=False):
        raw[str(row.ticker)] = max(float(row.composite_score), 0.0)
    return normalize_sleeve_weights(raw, list(valid["ticker"].astype(str)))


def _asset_current_weight(ticker: str, current_weights: dict[str, float]) -> float:
    return float(current_weights.get(str(ticker), 0.0))


def build_opaque_packet(
    day_df: pd.DataFrame,
    current_weights: dict[str, float],
    id_mapping: pd.DataFrame,
    decision_step: int = 0,
) -> dict[str, Any]:
    _ = decision_step
    valid = get_valid_universe(day_df).merge(id_mapping, on="ticker", how="left", suffixes=("", "_map"))
    if "asset_id_map" in valid.columns:
        valid["asset_id"] = valid["asset_id"].fillna(valid["asset_id_map"])
    assets = []
    for row in valid.itertuples(index=False):
        assets.append(
            {
                "asset_id": str(row.asset_id),
                "signal_1_rank": int(row.market_rank),
                "signal_1_score": float(row.market_score),
                "signal_2_rank": int(row.flow_rank),
                "signal_2_score": float(row.flow_score),
                "signal_3_rank": int(row.rotation_rank),
                "signal_3_score": float(row.rotation_score),
                "current_sleeve_weight": _asset_current_weight(str(row.ticker), current_weights),
            }
        )
    return {
        "constraints": {
            "rank_1_is_best": True,
            "higher_score_is_better": True,
            "sleeve_weight_sum": 1.0,
        },
        "assets": assets,
    }


def build_opaque_prompt(packet: dict[str, Any], top_k: int = 10) -> str:
    prompt = {
        "instructions": [
            "You are given three independent ranking signals.",
            "Rank 1 is best.",
            "Higher score is better.",
            "Use only the provided ranks and scores.",
            "Allocate 100% of the predefined asset sleeve across the provided asset_id universe.",
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


def call_llm(prompt: str, model: str | None = None, timeout: int | None = None) -> tuple[str | None, dict[str, Any]]:
    config = load_llm_config()
    provider = str(config["provider"])
    api_key = str(config.get("api_key", ""))
    model = model or str(config["model"])
    timeout = timeout or int(config["timeout_seconds"])
    if _is_placeholder_secret(api_key):
        return None, {"provider": "none", "model": model, "prompt_tokens": 0, "response_tokens": 0}
    payload = {
        "model": model,
        "temperature": float(config["temperature"]),
        "max_tokens": int(config["max_tokens"]),
        "messages": [{"role": "user", "content": prompt}],
    }
    if config.get("response_format"):
        payload["response_format"] = config["response_format"]
    if config.get("thinking"):
        payload["thinking"] = config["thinking"]
    if config.get("chat_template_kwargs"):
        payload["chat_template_kwargs"] = config["chat_template_kwargs"]
    request = urllib.request.Request(
        str(config["chat_completions_url"]),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    max_retries = max(1, int(config.get("max_retries", 1)))
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:1000]}"
            retryable_degraded = "DEGRADED function cannot be invoked" in body
            should_retry = (exc.code in {408, 429, 500, 502, 503, 504} or retryable_degraded) and '"1113"' not in body and '"1211"' not in body
            if not should_retry or attempt == max_retries:
                return None, {"provider": provider, "model": model, "error": last_error, "attempts": attempt}
            time.sleep(float(config.get("retry_sleep_seconds", 0.0)) * attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt == max_retries:
                return None, {"provider": provider, "model": model, "error": last_error, "attempts": attempt}
            time.sleep(float(config.get("retry_sleep_seconds", 0.0)) * attempt)
    else:
        return None, {"provider": provider, "model": model, "error": last_error, "attempts": max_retries}
    usage = data.get("usage", {})
    message = data["choices"][0].get("message", {})
    content = message.get("content")
    if content is None:
        content = ""
    success_sleep_seconds = max(0.0, float(config.get("success_sleep_seconds", 0.0)))
    if success_sleep_seconds > 0.0:
        time.sleep(success_sleep_seconds)
    return content, {
        "provider": provider,
        "model": model,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "response_tokens": int(usage.get("completion_tokens", 0)),
        "success_sleep_seconds": success_sleep_seconds,
    }


def classify_llm_failure(call_meta: dict[str, Any], validation_errors: list[str]) -> str:
    llm_config = load_llm_config()
    if _is_placeholder_secret(str(llm_config.get("api_key", ""))):
        return "missing_api_key"
    error = str(call_meta.get("error", ""))
    lowered = error.lower()
    if "degraded function cannot be invoked" in lowered:
        return "llm_degraded"
    if "http 429" in lowered or "too many requests" in lowered:
        return "llm_rate_limited"
    if "timed out" in lowered or "timeout" in lowered:
        return "llm_timeout"
    if error:
        return "llm_call_failed"
    if validation_errors:
        return "invalid_llm_output"
    return "llm_failed"


def parse_llm_json(response: str | None) -> dict[str, Any] | None:
    if not response:
        return None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        start = response.find("{")
        end = response.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def validate_llm_sleeve_weights(
    parsed: dict[str, Any] | None,
    valid_asset_ids: set[str],
    top_k: int,
    require_reason_codes: bool = False,
) -> tuple[bool, dict[str, float], list[str]]:
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return False, {}, ["response_not_json_object"]
    rows = parsed.get("target_weights")
    if isinstance(rows, dict):
        rows = [{"asset_id": asset_id, "sleeve_weight": weight} for asset_id, weight in rows.items()]
    if not isinstance(rows, list):
        return False, {}, ["missing_target_weights"]
    weights: dict[str, float] = {}
    for item in rows:
        if not isinstance(item, dict):
            errors.append("target_weight_not_object")
            continue
        asset_id = item.get("asset_id")
        if asset_id not in valid_asset_ids:
            errors.append("invalid_asset_id")
            continue
        try:
            weight = float(item.get("sleeve_weight"))
        except (TypeError, ValueError):
            errors.append("non_numeric_weight")
            continue
        if weight < 0 or not math.isfinite(weight):
            errors.append("invalid_weight")
            continue
        if require_reason_codes and "reason_codes" in item:
            bad_codes = set(item["reason_codes"]) - REASON_CODE_ENUM if isinstance(item["reason_codes"], list) else {"bad"}
            if bad_codes:
                errors.append("invalid_reason_codes")
        weights[str(asset_id)] = weights.get(str(asset_id), 0.0) + weight
    nonzero = sum(1 for weight in weights.values() if weight > 1e-12)
    max_nonzero = min(top_k, len(valid_asset_ids)) if top_k < 10 else len(valid_asset_ids)
    total = sum(weights.values())
    if nonzero > max_nonzero:
        errors.append("too_many_nonzero_weights")
    if abs(total - 1.0) > 1e-6:
        errors.append("weights_do_not_sum_to_one")
    if errors:
        return False, weights, errors
    return True, weights, []


def repair_invalid_llm_output_once(
    parsed: dict[str, Any] | None,
    valid_asset_ids: set[str],
    top_k: int,
) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(parsed, dict):
        return None, False
    rows = parsed.get("target_weights")
    if isinstance(rows, dict):
        rows = [{"asset_id": asset_id, "sleeve_weight": weight} for asset_id, weight in rows.items()]
    if not isinstance(rows, list):
        return None, False
    weights: dict[str, float] = {}
    for item in rows:
        if not isinstance(item, dict) or item.get("asset_id") not in valid_asset_ids:
            continue
        try:
            weight = max(float(item.get("sleeve_weight", 0.0)), 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        weights[str(item["asset_id"])] = weights.get(str(item["asset_id"]), 0.0) + weight
    if top_k < 10:
        keep = {asset for asset, _ in sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:top_k]}
        weights = {asset: (weight if asset in keep else 0.0) for asset, weight in weights.items()}
    total = sum(weights.values())
    if total <= 0.0:
        return None, False
    repaired = {
        "target_weights": [
            {"asset_id": asset_id, "sleeve_weight": weight / total}
            for asset_id, weight in sorted(weights.items())
            if weight > 0.0
        ],
    }
    return repaired, True


def map_asset_id_to_ticker(weights: dict[str, float], id_mapping: pd.DataFrame) -> dict[str, float]:
    asset_to_ticker = dict(zip(id_mapping["asset_id"].astype(str), id_mapping["ticker"].astype(str)))
    return {asset_to_ticker[asset_id]: weight for asset_id, weight in weights.items() if asset_id in asset_to_ticker}


def generate_A2a_weight(
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
    packet = build_opaque_packet(day_df, current_weights, id_mapping, decision_step=decision_step)
    prompt = build_opaque_prompt(packet, top_k=top_k)
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


def _current_sleeve_weights(state: dict[str, Any], day_df: pd.DataFrame) -> dict[str, float]:
    quantities = state.get("quantities", {})
    if not quantities:
        return {}
    close_by_ticker = dict(zip(day_df["ticker"].astype(str), day_df["close"].astype(float)))
    values = {ticker: qty * close_by_ticker.get(ticker, 0.0) for ticker, qty in quantities.items()}
    total = sum(values.values())
    return {ticker: value / total for ticker, value in values.items()} if total > 0.0 else {}


def _floor_quantity(value: float, price: float) -> float:
    if price <= 0.0 or value <= 0.0 or not math.isfinite(value) or not math.isfinite(price):
        return 0.0
    return float(math.floor(value / price))


def rebalance_and_mark_to_market_next_day(
    state: dict[str, Any],
    target_weights: dict[str, float],
    price_t: pd.DataFrame,
    price_t1: pd.DataFrame,
    stock_ratio: float = STOCK_RATIO,
) -> tuple[dict[str, Any], dict[str, float]]:
    open_next = dict(zip(price_t1["ticker"].astype(str), price_t1["open"].astype(float)))
    close_next = dict(zip(price_t1["ticker"].astype(str), price_t1["close"].astype(float)))
    close_today = dict(zip(price_t["ticker"].astype(str), price_t["close"].astype(float)))
    quantities = state.get("quantities", {})
    cash = float(state.get("cash", INITIAL_NAV))
    nav_prev = float(state.get("nav", INITIAL_NAV))
    nav_open_pre = cash + sum(qty * open_next.get(ticker, close_today.get(ticker, 0.0)) for ticker, qty in quantities.items())
    old_values = {ticker: quantities.get(ticker, 0.0) * open_next.get(ticker, 0.0) for ticker in set(quantities) | set(target_weights)}
    commission = 0.0
    nav_open_post = nav_open_pre
    target_values: dict[str, float] = {}
    new_quantities: dict[str, float] = {}
    for _ in range(50):
        target_budget = {ticker: stock_ratio * nav_open_post * weight for ticker, weight in target_weights.items()}
        new_quantities = {
            ticker: _floor_quantity(value, open_next[ticker])
            for ticker, value in target_budget.items()
            if ticker in open_next and open_next[ticker] > 0.0 and value > 0.0
        }
        target_values = {ticker: qty * open_next[ticker] for ticker, qty in new_quantities.items()}
        next_commission = COMMISSION_RATE * sum(abs(target_values.get(ticker, 0.0) - old_values.get(ticker, 0.0)) for ticker in set(target_values) | set(old_values))
        next_nav_open_post = nav_open_pre - next_commission
        if abs(next_commission - commission) <= 1e-10:
            commission = next_commission
            nav_open_post = next_nav_open_post
            break
        commission = next_commission
        nav_open_post = next_nav_open_post
    target_budget = {ticker: stock_ratio * nav_open_post * weight for ticker, weight in target_weights.items()}
    new_quantities = {
        ticker: _floor_quantity(value, open_next[ticker])
        for ticker, value in target_budget.items()
        if ticker in open_next and open_next[ticker] > 0.0 and value > 0.0
    }
    target_values = {ticker: qty * open_next[ticker] for ticker, qty in new_quantities.items()}
    cash_post = nav_open_post - sum(target_values.values())
    nav_close = cash_post + sum(qty * close_next.get(ticker, 0.0) for ticker, qty in new_quantities.items())
    turnover_value = sum(abs(target_values.get(ticker, 0.0) - old_values.get(ticker, 0.0)) for ticker in set(target_values) | set(old_values))
    new_state = {"nav": nav_close, "cash": cash_post, "quantities": new_quantities}
    result = {
        "nav_close": nav_close,
        "nav_open_pre": nav_open_pre,
        "nav_open_post": nav_open_post,
        "overnight_pnl": nav_open_pre - nav_prev,
        "open_to_close_pnl": nav_close - nav_open_post,
        "commission": commission,
        "turnover_value": turnover_value,
        "turnover_ratio": turnover_value / nav_open_pre if nav_open_pre > 0.0 else 0.0,
        "daily_return": nav_close / nav_prev - 1.0 if nav_prev > 0.0 else 0.0,
    }
    return new_state, result


def compute_performance_metrics(daily_results: pd.DataFrame, fallback_count: int) -> dict[str, Any]:
    if daily_results.empty:
        return {"final_nav": INITIAL_NAV, "fallback_rate": 0.0}
    returns = daily_results["daily_return"].astype(float)
    final_nav = float(daily_results["nav_close"].iloc[-1])
    total_return = final_nav / INITIAL_NAV - 1.0
    years = max(len(daily_results) / 252.0, 1 / 252.0)
    annual_vol = float(returns.std(ddof=0) * math.sqrt(252)) if len(returns) else 0.0
    cagr = float((final_nav / INITIAL_NAV) ** (1 / years) - 1.0)
    sharpe = float((returns.mean() * 252) / annual_vol) if annual_vol > 0 else 0.0
    downside = returns[returns < 0]
    sortino_den = float(downside.std(ddof=0) * math.sqrt(252)) if len(downside) else 0.0
    wealth = daily_results["nav_close"].astype(float)
    max_dd = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "final_nav": final_nav,
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": float((returns.mean() * 252) / sortino_den) if sortino_den > 0 else 0.0,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else 0.0,
        "hit_ratio": float((returns > 0).mean()),
        "avg_turnover_value": float(daily_results["turnover_value"].mean()),
        "avg_turnover_ratio": float(daily_results["turnover_ratio"].mean()),
        "turnover": float(daily_results["turnover_ratio"].mean()),
        "total_commission": float(daily_results["commission"].sum()),
        "commission_drag": float(daily_results["commission"].sum() / INITIAL_NAV),
        "fallback_rate": fallback_count / len(daily_results),
    }


def run_llm_backtest(
    panel: pd.DataFrame,
    strategy: str,
    top_k: int,
    out_dir: str | Path,
    generator,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    validate_top_k(top_k)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    id_mapping = create_id_mapping(panel)
    dates = sorted(panel["date"].dropna().unique())
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None
    regime_path = default_regime_path()
    regime = load_regime_table(regime_path)
    daily_path = out_path / "daily_results.csv"
    weights_path = out_path / "weights.csv"
    trade_path = out_path / "trade_history.csv"
    log_path = out_path / "decision_log.jsonl"
    checkpoint_path = out_path / "checkpoint_state.json"
    state: dict[str, Any] = {"nav": INITIAL_NAV, "cash": INITIAL_NAV, "quantities": {}}
    daily_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    fallback_count = 0
    completed_dates: set[str] = set()
    fail_on_fallback = _env_flag("LLM_FAIL_ON_FALLBACK", False)
    progress_interval = max(1, int(os.getenv("LLM_PROGRESS_INTERVAL", "25")))

    if checkpoint_path.exists() and daily_path.exists() and weights_path.exists() and log_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("strategy") == strategy and int(checkpoint.get("top_k", -1)) == int(top_k):
            state = checkpoint.get("state", state)
            state["quantities"] = {str(k): float(v) for k, v in state.get("quantities", {}).items()}
            daily_rows = read_csv_records(daily_path)
            weight_rows = read_csv_records(weights_path)
            if trade_path.exists():
                trade_rows = read_csv_records(trade_path)
            with log_path.open("r", encoding="utf-8") as handle:
                log_rows = [json.loads(line) for line in handle if line.strip()]
            completed_dates = {str(row["decision_date"]) for row in daily_rows}
            fallback_count = sum(1 for row in daily_rows if bool(row.get("fallback_used")))

    def persist_checkpoint(complete: bool = False) -> None:
        write_csv_records(daily_path, daily_rows, DAILY_RESULT_COLUMNS)
        write_csv_records(weights_path, weight_rows, WEIGHT_COLUMNS)
        write_csv_records(trade_path, trade_rows, TRADE_COLUMNS)
        with log_path.open("w", encoding="utf-8") as handle:
            for row in log_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        checkpoint_path.write_text(
            json.dumps(
                {
                    "strategy": strategy,
                    "top_k": top_k,
                    "complete": complete,
                    "completed_decisions": len(completed_dates),
                    "state": state,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    for step, decision_date in enumerate(dates[:-1]):
        decision_ts = pd.Timestamp(decision_date)
        if start_ts is not None and decision_ts < start_ts:
            continue
        if end_ts is not None and decision_ts > end_ts:
            continue
        decision_date_text = pd.Timestamp(decision_date).date().isoformat()
        if decision_date_text in completed_dates:
            continue
        execution_date = dates[step + 1]
        regime_info = regime_info_for_date(regime, pd.Timestamp(execution_date))
        day_df = apply_regime_market_score(panel[panel["date"] == decision_date], regime_info)
        next_df = panel[panel["date"] == execution_date].copy()
        if get_valid_universe(day_df).empty:
            continue
        current_weights = _current_sleeve_weights(state, day_df)
        decision = generator(day_df, current_weights, top_k=top_k, decision_step=step, id_mapping=id_mapping)
        target_weights = decision.weights
        if fail_on_fallback and decision.fallback_used:
            failure_log = {
                "strategy": strategy,
                "top_k": top_k,
                "decision_date": decision_date_text,
                "decision_step": step,
                **decision.log,
            }
            log_rows.append(failure_log)
            persist_checkpoint()
            raise RuntimeError(
                "LLM fallback blocked by LLM_FAIL_ON_FALLBACK=1: "
                + json.dumps(failure_log, ensure_ascii=False, sort_keys=True, default=str)
            )
        fallback_count += int(decision.fallback_used)
        execution_stock_ratio = regime_stock_ratio_for_date(regime, execution_date, STOCK_RATIO)
        old_quantities = {str(k): float(v) for k, v in state.get("quantities", {}).items()}
        state, result = rebalance_and_mark_to_market_next_day(state, target_weights, day_df, next_df, execution_stock_ratio)
        trade_rows.extend(
            build_trade_rows(
                strategy=strategy,
                decision_date=decision_date_text,
                execution_date=pd.Timestamp(execution_date).date().isoformat(),
                decision_df=day_df,
                execution_df=next_df,
                old_qty=old_quantities,
                new_qty=state.get("quantities", {}),
                target_sleeve_weights=target_weights,
                result=result,
                stock_ratio=execution_stock_ratio,
            )
        )
        daily_rows.append(
            {
                "decision_date": decision_date_text,
                "execution_date": pd.Timestamp(execution_date).date().isoformat(),
                "strategy": strategy,
                **result,
                "stock_ratio": execution_stock_ratio,
                "fallback_used": decision.fallback_used,
                "effective_num_positions": sum(1 for weight in target_weights.values() if weight > 1e-12),
            }
        )
        etf_by_ticker = dict(zip(day_df["ticker"].astype(str), day_df.get("ETF_id", day_df["ticker"]).astype(str)))
        for ticker, weight in target_weights.items():
            weight_rows.append(
                {
                    "decision_date": decision_date_text,
                    "execution_date": pd.Timestamp(execution_date).date().isoformat(),
                    "ticker": ticker,
                    "ETF_id": etf_by_ticker.get(ticker, ticker),
                    "sleeve_weight": weight,
                    "portfolio_weight": weight * execution_stock_ratio,
                }
            )
        log_rows.append(
            {
                "strategy": strategy,
                "top_k": top_k,
                **decision.log,
            }
        )
        completed_dates.add(decision_date_text)
        persist_checkpoint()
        if len(daily_rows) % progress_interval == 0:
            print(
                f"PROGRESS {strategy} top_k={top_k} decisions={len(daily_rows)} "
                f"latest_decision={decision_date_text} fallback_count={fallback_count}",
                flush=True,
            )
    daily_df = pd.DataFrame(daily_rows)
    last_decision_ts = pd.Timestamp(dates[-2]) if len(dates) >= 2 else None
    is_complete = end_ts is None or (last_decision_ts is not None and end_ts >= last_decision_ts)
    persist_checkpoint(complete=is_complete)
    llm_config = load_llm_config()
    summary = {
        "strategy": strategy,
        "top_k": top_k,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "llm_provider": str(llm_config["provider"]) if not _is_placeholder_secret(str(llm_config.get("api_key", ""))) else "none",
        "model_id": str(llm_config["model"]),
        "commission_rate": COMMISSION_RATE,
        "stock_ratio": STOCK_RATIO,
        "stock_ratio_mode": "regime" if regime is not None else "constant",
        "regime_path": str(regime_path) if regime_path else None,
        "initial_nav": INITIAL_NAV,
        **compute_performance_metrics(daily_df, fallback_count),
    }
    with (out_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)


REASON_CODE_ENUM = {
    "rank_alignment_strong",
    "market_rank_strong",
    "flow_rank_strong",
    "rotation_rank_strong",
    "rank_conflict",
    "market_rank_weak",
    "flow_rank_weak",
    "rotation_rank_weak",
    "turnover_control",
    "concentration_control",
    "constraints_satisfied",
    "fallback_required",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A2a LLM opaque rank allocator.")
    parser.add_argument("--rank-panel", default="rank_panel.csv")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--out-root", default="out")
    parser.add_argument("--start-date", default=None, help="Optional inclusive decision-date start.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive decision-date end.")
    args = parser.parse_args()
    panel = load_rank_panel(args.rank_panel)
    run_llm_backtest(
        panel,
        "A2a_LLM_OPAQUE",
        args.top_k,
        Path(args.out_root) / f"A2a_k{args.top_k}",
        generate_A2a_weight,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TESTING_DIR = PROJECT_DIR / "testing"

PRICE_COLUMNS = [
    "date",
    "ticker",
    "name",
    "gics_sector",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "index_close",
]

FLOW_COLUMNS = [
    "date",
    "ticker",
    "individual_net_buy",
    "foreign_net_buy",
    "institution_net_buy",
    "finance_invest_net_buy",
    "insurance_net_buy",
    "trust_net_buy",
    "other_finance_net_buy",
    "bank_net_buy",
    "pension_net_buy",
    "private_fund_net_buy",
    "nation_net_buy",
    "other_corp_net_buy",
    "other_foreign_net_buy",
]

FLOW_FIELD_MAP = {
    "individual_net_buy": ["ind_invsr", "individual", "ind"],
    "foreign_net_buy": ["frgnr_invsr", "foreign", "frgnr"],
    "institution_net_buy": ["orgn", "institution", "inst"],
    "finance_invest_net_buy": ["fnnc_invt"],
    "insurance_net_buy": ["insrnc"],
    "trust_net_buy": ["invtrt", "trust"],
    "other_finance_net_buy": ["etc_fnnc"],
    "bank_net_buy": ["bank"],
    "pension_net_buy": ["penfnd_etc", "pension"],
    "private_fund_net_buy": ["samo_fund"],
    "nation_net_buy": ["natn"],
    "other_corp_net_buy": ["etc_corp"],
    "other_foreign_net_buy": ["natfor"],
}


def load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.split("#", 1)[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def clean_number(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return 0.0
    sign = -1.0 if text.startswith("-") else 1.0
    text = re.sub(r"[^0-9.]", "", text)
    return sign * float(text) if text else 0.0


def normalize_date(value: Any) -> str | None:
    text = re.sub(r"[^0-9]", "", str(value or ""))
    if len(text) >= 8:
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return None


def first_value(row: dict[str, Any], names: list[str]) -> Any:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


class KiwoomRestClient:
    def __init__(self, base_url: str, app_key: str, secret_key: str, token: str = "", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.app_key = app_key
        self.secret_key = secret_key
        self.token = token
        self.timeout = timeout

    def ensure_token(self) -> str:
        if self.token:
            return self.token
        if not self.app_key or not self.secret_key:
            raise RuntimeError("Set KIWOOM_APP_KEY and KIWOOM_SECRET_KEY, or set KIWOOM_TOKEN.")
        body = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.secret_key}
        data = self._post("/oauth2/token", body, headers={})
        token = data.get("token") or data.get("access_token")
        if not token:
            raise RuntimeError(f"Kiwoom token response did not include token: {data}")
        self.token = str(token)
        return self.token

    def call_tr(
        self,
        endpoint: str,
        api_id: str,
        body: dict[str, Any],
        max_pages: int | None = None,
        stop_when: Callable[[list[dict[str, Any]]], bool] | None = None,
    ) -> list[dict[str, Any]]:
        token = self.ensure_token()
        rows: list[dict[str, Any]] = []
        cont_yn = "N"
        next_key = ""
        max_pages = max(1, int(max_pages if max_pages is not None else os.getenv("KIWOOM_MAX_PAGES", "1")))
        page = 0
        while True:
            page += 1
            headers = {"authorization": f"Bearer {token}", "api-id": api_id}
            if cont_yn == "Y":
                headers["cont-yn"] = cont_yn
                headers["next-key"] = next_key
            response = self._post(endpoint, body, headers=headers, include_headers=True)
            payload, response_headers = response
            rows.extend(extract_records(payload))
            cont_yn = str(response_headers.get("cont-yn", response_headers.get("Cont-Yn", "N"))).upper()
            next_key = str(response_headers.get("next-key", response_headers.get("Next-Key", "")))
            if stop_when is not None and stop_when(rows):
                break
            if page >= max_pages or cont_yn != "Y" or not next_key:
                break
            time.sleep(float(os.getenv("KIWOOM_CONT_SLEEP_SECONDS", "0.2")))
        return rows

    def _post(
        self,
        endpoint: str,
        body: dict[str, Any],
        headers: dict[str, str],
        include_headers: bool = False,
    ) -> Any:
        max_retries = max(1, int(os.getenv("KIWOOM_MAX_RETRIES", "5")))
        retry_sleep = max(0.0, float(os.getenv("KIWOOM_RETRY_SLEEP_SECONDS", "10")))
        last_error = ""
        for attempt in range(1, max_retries + 1):
            request = urllib.request.Request(
                self.base_url + endpoint,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json;charset=UTF-8", **headers},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    if include_headers:
                        return payload, dict(response.headers.items())
                    return payload
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = f"Kiwoom HTTP {exc.code}: {detail[:1000]}"
                if exc.code != 429 or attempt == max_retries:
                    raise RuntimeError(last_error) from exc
                time.sleep(retry_sleep * attempt)
        raise RuntimeError(last_error or "Kiwoom request failed")


def extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    preferred = [
        "stk_dt_pole_chart_qry",
        "stk_invsr_orgn_chart",
        "stk_invsr_orgn",
        "inds_cur_prc_daly",
        "output",
        "data",
    ]
    for key in preferred:
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    for value in payload.values():
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return value
    return [payload]


def load_baseline(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype={"ticker": "string"})
        if "index_close" in columns and "index_close" not in df.columns and "kospi_close" in df.columns:
            df["index_close"] = df["kospi_close"]
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        return df[columns]
    return pd.DataFrame(columns=columns)


def ticker_metadata(price_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    if price_df.empty:
        return meta
    for ticker, group in price_df.dropna(subset=["ticker"]).groupby("ticker"):
        row = group.sort_values("date").iloc[-1]
        meta[str(ticker)] = {"name": str(row.get("name", ticker)), "gics_sector": str(row.get("gics_sector", ""))}
    return meta


def parse_price_rows(
    ticker: str,
    rows: list[dict[str, Any]],
    meta: dict[str, dict[str, str]],
    amount_divisor: float,
) -> list[dict[str, Any]]:
    parsed = []
    for row in rows:
        date = normalize_date(first_value(row, ["dt", "date", "trd_dt", "stck_bsop_date"]))
        if not date:
            continue
        close = clean_number(first_value(row, ["cur_prc", "close", "clpr", "stck_clpr", "trd_prc"]))
        open_ = clean_number(first_value(row, ["open_pric", "open_prc", "open", "stck_oprc"]))
        high = clean_number(first_value(row, ["high_pric", "high_prc", "high", "stck_hgpr"]))
        low = clean_number(first_value(row, ["low_pric", "low_prc", "low", "stck_lwpr"]))
        volume = clean_number(first_value(row, ["trde_qty", "acc_trdvol", "volume", "vol"]))
        amount_raw = clean_number(first_value(row, ["acc_trde_prica", "trde_prica", "acc_trdval", "amount", "amt"]))
        info = meta.get(str(ticker), {"name": str(ticker), "gics_sector": ""})
        parsed.append(
            {
                "date": date,
                "ticker": str(ticker),
                "name": info["name"],
                "gics_sector": info["gics_sector"],
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount_raw / amount_divisor if amount_divisor else amount_raw,
                "index_close": pd.NA,
            }
        )
    return parsed


def parse_flow_rows(ticker: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed = []
    for row in rows:
        date = normalize_date(first_value(row, ["dt", "date", "trd_dt"]))
        if not date:
            continue
        out = {"date": date, "ticker": str(ticker)}
        for target_col, aliases in FLOW_FIELD_MAP.items():
            out[target_col] = clean_number(first_value(row, aliases))
        parsed.append(out)
    return parsed


def parse_index_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    parsed = {}
    for row in rows:
        date = normalize_date(first_value(row, ["dt", "dt_n", "date", "trd_dt"]))
        close = abs(clean_number(first_value(row, ["cur_prc", "cur_prc_n", "close", "clpr", "inds_clpr"])))
        if date and close > 0:
            parsed[date] = close
    return parsed


def index_column_for_price_frame(df: pd.DataFrame) -> str:
    if "index_close" in df.columns:
        return "index_close"
    if "kospi_close" in df.columns:
        return "kospi_close"
    raise ValueError("price CSV must contain index_close or kospi_close")


def selected_price_dates(df: pd.DataFrame, from_date: str | None, to_date: str | None) -> list[str]:
    dates = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    mask = pd.Series(True, index=df.index)
    if from_date:
        mask &= dates >= from_date
    if to_date:
        mask &= dates <= to_date
    return sorted(dates.loc[mask].dropna().unique())


def validate_index_series(index_by_date: dict[str, float], required_dates: list[str]) -> dict[str, Any]:
    missing = [date for date in required_dates if date not in index_by_date]
    if missing:
        sample = missing[:10]
        raise ValueError(f"KOSPI index fetch missing {len(missing)} required dates, examples={sample}")

    values = pd.Series({date: float(index_by_date[date]) for date in required_dates}).sort_index()
    if (values <= 0).any():
        bad = values.loc[values <= 0].head(10).to_dict()
        raise ValueError(f"KOSPI index close must be positive: {bad}")

    min_close = float(os.getenv("KIWOOM_INDEX_MIN_CLOSE", "1000"))
    max_close = float(os.getenv("KIWOOM_INDEX_MAX_CLOSE", "20000"))
    out_of_scale = values.loc[(values < min_close) | (values > max_close)]
    if not out_of_scale.empty:
        sample = out_of_scale.head(10).to_dict()
        raise ValueError(
            f"KOSPI index close outside expected [{min_close}, {max_close}] range: {sample}. "
            "Set KIWOOM_INDEX_MIN_CLOSE/KIWOOM_INDEX_MAX_CLOSE only after confirming the API scale."
        )

    log_returns = np_log_safe(values.astype(float) / values.astype(float).shift(1))
    abnormal = log_returns.loc[log_returns.abs() > float(os.getenv("KIWOOM_INDEX_MAX_ABS_LOG_RETURN", "0.15"))]
    if not abnormal.empty:
        sample = abnormal.head(10).to_dict()
        raise ValueError(f"KOSPI index close has abnormal daily log returns: {sample}")

    return {
        "date_count": int(len(values)),
        "min_date": str(values.index.min()),
        "max_date": str(values.index.max()),
        "min_close": float(values.min()),
        "max_close": float(values.max()),
    }


def np_log_safe(values: pd.Series) -> pd.Series:
    import numpy as np

    values = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.log(values), index=values.index).replace([np.inf, -np.inf], pd.NA).dropna()


def overwrite_price_index_column(path: Path, index_by_date: dict[str, float], from_date: str | None, to_date: str | None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"ticker": "string"})
    index_col = index_column_for_price_frame(df)
    df[index_col] = pd.to_numeric(df[index_col], errors="coerce").astype(float)
    date_text = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    required_dates = selected_price_dates(df, from_date, to_date)
    validation = validate_index_series(index_by_date, required_dates)
    mask = date_text.isin(required_dates)
    df.loc[mask, index_col] = date_text.loc[mask].map(index_by_date).astype(float)
    per_date_unique = df.loc[mask].groupby(date_text.loc[mask])[index_col].nunique(dropna=False)
    if (per_date_unique != 1).any():
        bad = per_date_unique.loc[per_date_unique != 1].head(10).to_dict()
        raise ValueError(f"KOSPI index close is not identical across tickers for dates: {bad}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "path": str(path),
        "index_column": index_col,
        "rows_updated": int(mask.sum()),
        **validation,
    }


def fetch_index_map(client: KiwoomRestClient, from_date: str | None, to_date: str | None) -> dict[str, float]:
    index_endpoint = os.getenv("KIWOOM_INDEX_ENDPOINT", "/api/dostk/sect")
    index_api_id = os.getenv("KIWOOM_INDEX_API_ID", "ka20009")
    kospi_code = os.getenv("KIWOOM_KOSPI_CODE", "001")
    index_body = {
        "mrkt_tp": os.getenv("KIWOOM_INDEX_MARKET_TP", "0"),
        "inds_cd": kospi_code,
        "base_dt": (to_date or "").replace("-", ""),
    }
    max_pages = int(os.getenv("KIWOOM_INDEX_MAX_PAGES", os.getenv("KIWOOM_MAX_PAGES", "200")))
    def reached_from_date(rows: list[dict[str, Any]]) -> bool:
        if not from_date:
            return False
        parsed = parse_index_rows(rows)
        return bool(parsed) and min(parsed) <= from_date

    return parse_index_rows(
        client.call_tr(index_endpoint, index_api_id, index_body, max_pages=max_pages, stop_when=reached_from_date)
    )


def run_index_only(args: argparse.Namespace) -> None:
    target_paths = [args.price_out, *args.extra_price_out]
    existing_paths = [path for path in target_paths if path.exists()]
    if not existing_paths:
        raise FileNotFoundError("At least one --price-out/--extra-price-out CSV must exist for --index-only")
    all_dates: list[str] = []
    for path in existing_paths:
        all_dates.extend(selected_price_dates(pd.read_csv(path, dtype={"ticker": "string"}), None, None))
    all_dates = sorted(set(all_dates))
    from_date = args.from_date or all_dates[0]
    to_date = args.to_date or all_dates[-1]

    use_mock = os.getenv("KIWOOM_USE_MOCK", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    default_base_url = "https://mockapi.kiwoom.com" if use_mock else "https://api.kiwoom.com"
    dry_run_payload = {
        "mode": "index_only",
        "from_date": from_date,
        "to_date": to_date,
        "targets": [str(path) for path in target_paths],
        "base_url": os.getenv("KIWOOM_BASE_URL", default_base_url),
        "index_endpoint": os.getenv("KIWOOM_INDEX_ENDPOINT", "/api/dostk/sect"),
        "index_api_id": os.getenv("KIWOOM_INDEX_API_ID", "ka20009"),
        "kospi_code": os.getenv("KIWOOM_KOSPI_CODE", "001"),
        "index_max_pages": int(os.getenv("KIWOOM_INDEX_MAX_PAGES", os.getenv("KIWOOM_MAX_PAGES", "200"))),
    }
    if args.dry_run:
        print(json.dumps(dry_run_payload, ensure_ascii=False, indent=2))
        return

    client = KiwoomRestClient(
        base_url=str(dry_run_payload["base_url"]),
        app_key=os.getenv("KIWOOM_APP_KEY", ""),
        secret_key=os.getenv("KIWOOM_SECRET_KEY", ""),
        token=os.getenv("KIWOOM_TOKEN", ""),
        timeout=int(os.getenv("KIWOOM_TIMEOUT_SECONDS", "30")),
    )
    index_map = fetch_index_map(client, from_date, to_date)
    summaries = [
        overwrite_price_index_column(path, index_map, from_date, to_date)
        for path in target_paths
    ]
    print(json.dumps({"index_rows_fetched": len(index_map), "targets": summaries}, ensure_ascii=False, indent=2))


def filter_dates(rows: list[dict[str, Any]], from_date: str | None, to_date: str | None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        date = str(row.get("date", ""))
        if from_date and date < from_date:
            continue
        if to_date and date > to_date:
            continue
        out.append(row)
    return out


def merge_and_save(base: pd.DataFrame, new_rows: list[dict[str, Any]], path: Path, columns: list[str]) -> pd.DataFrame:
    additions = pd.DataFrame(new_rows)
    merged = pd.concat([base, additions], ignore_index=True) if not additions.empty else base.copy()
    if merged.empty:
        merged = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[columns].drop_duplicates(["date", "ticker"], keep="last").sort_values(["date", "ticker"])
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Kiwoom REST data and append pilot price/flow CSVs.")
    parser.add_argument("--from-date", help="YYYY-MM-DD inclusive")
    parser.add_argument("--to-date", help="YYYY-MM-DD inclusive")
    parser.add_argument("--tickers", nargs="*", help="ETF tickers. Default uses baseline price CSV tickers.")
    parser.add_argument("--baseline-price", type=Path, default=TESTING_DIR / "sector_all_merged.csv")
    parser.add_argument("--baseline-flow", type=Path, default=TESTING_DIR / "sector_fund_flow.csv")
    parser.add_argument("--price-out", type=Path, default=BASE_DIR / "data" / "sector_all_merged.csv")
    parser.add_argument("--extra-price-out", type=Path, action="append", default=[], help="Additional price CSV to update in --index-only mode.")
    parser.add_argument("--flow-out", type=Path, default=BASE_DIR / "data" / "sector_fund_flow.csv")
    parser.add_argument("--amount-divisor", type=float, default=float(os.getenv("KIWOOM_AMOUNT_DIVISOR", "1")))
    parser.add_argument("--index-only", action="store_true", help="Fetch only KOSPI index closes and overwrite price CSV index columns.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.index_only:
        run_index_only(args)
        return

    price_base = load_baseline(args.price_out if args.price_out.exists() else args.baseline_price, PRICE_COLUMNS)
    flow_base = load_baseline(args.flow_out if args.flow_out.exists() else args.baseline_flow, FLOW_COLUMNS)
    tickers = args.tickers or sorted(price_base["ticker"].dropna().astype(str).unique())
    if not tickers:
        raise SystemExit("No tickers available. Pass --tickers or provide a baseline price CSV.")

    if args.dry_run:
        print(json.dumps({"tickers": tickers, "price_out": str(args.price_out), "flow_out": str(args.flow_out)}, indent=2))
        return

    use_mock = os.getenv("KIWOOM_USE_MOCK", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    default_base_url = "https://mockapi.kiwoom.com" if use_mock else "https://api.kiwoom.com"
    client = KiwoomRestClient(
        base_url=os.getenv("KIWOOM_BASE_URL", default_base_url),
        app_key=os.getenv("KIWOOM_APP_KEY", ""),
        secret_key=os.getenv("KIWOOM_SECRET_KEY", ""),
        token=os.getenv("KIWOOM_TOKEN", ""),
        timeout=int(os.getenv("KIWOOM_TIMEOUT_SECONDS", "30")),
    )
    price_endpoint = os.getenv("KIWOOM_PRICE_ENDPOINT", "/api/dostk/chart")
    price_api_id = os.getenv("KIWOOM_PRICE_API_ID", "ka10081")
    flow_endpoint = os.getenv("KIWOOM_FLOW_ENDPOINT", "/api/dostk/chart")
    flow_api_id = os.getenv("KIWOOM_FLOW_API_ID", "ka10060")
    index_endpoint = os.getenv("KIWOOM_INDEX_ENDPOINT", "/api/dostk/sect")
    index_api_id = os.getenv("KIWOOM_INDEX_API_ID", "ka20009")
    kospi_code = os.getenv("KIWOOM_KOSPI_CODE", "001")

    meta = ticker_metadata(price_base)
    price_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        price_body = {
            "stk_cd": str(ticker),
            "base_dt": (args.to_date or "").replace("-", ""),
            "upd_stkpc_tp": os.getenv("KIWOOM_PRICE_UPD_STKPC_TP", "1"),
        }
        raw_price = client.call_tr(price_endpoint, price_api_id, price_body)
        price_rows.extend(filter_dates(parse_price_rows(str(ticker), raw_price, meta, args.amount_divisor), args.from_date, args.to_date))

        flow_body = {
            "dt": (args.to_date or "").replace("-", ""),
            "stk_cd": str(ticker),
            "amt_qty_tp": os.getenv("KIWOOM_FLOW_AMOUNT_QTY_TP", "1"),
            "trde_tp": os.getenv("KIWOOM_FLOW_TRADE_TP", "0"),
            "unit_tp": os.getenv("KIWOOM_FLOW_UNIT_TP", "1000"),
        }
        raw_flow = client.call_tr(flow_endpoint, flow_api_id, flow_body)
        flow_rows.extend(filter_dates(parse_flow_rows(str(ticker), raw_flow), args.from_date, args.to_date))
        time.sleep(float(os.getenv("KIWOOM_REQUEST_SLEEP_SECONDS", "1.0")))

    index_body = {
        "mrkt_tp": os.getenv("KIWOOM_INDEX_MARKET_TP", "0"),
        "inds_cd": kospi_code,
        "base_dt": (args.to_date or "").replace("-", ""),
    }
    index_map = parse_index_rows(client.call_tr(index_endpoint, index_api_id, index_body))
    if index_map:
        for row in price_rows:
            row["index_close"] = index_map.get(row["date"], row.get("index_close", pd.NA))

    existing_index = price_base.dropna(subset=["index_close"]).drop_duplicates("date").set_index("date")["index_close"].to_dict()
    for row in price_rows:
        if pd.isna(row.get("index_close")):
            row["index_close"] = existing_index.get(row["date"], pd.NA)

    price_df = merge_and_save(price_base, price_rows, args.price_out, PRICE_COLUMNS)
    flow_df = merge_and_save(flow_base, flow_rows, args.flow_out, FLOW_COLUMNS)
    print(
        json.dumps(
            {
                "price_rows_fetched": len(price_rows),
                "flow_rows_fetched": len(flow_rows),
                "price_rows_total": len(price_df),
                "flow_rows_total": len(flow_df),
                "price_out": str(args.price_out),
                "flow_out": str(args.flow_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

# ETF Rank Backtest and Pilot System

이 저장소는 한국 섹터 ETF 유니버스를 대상으로 rank 기반 투자 전략을 만들고, 백테스트와 파일럿 운용 시뮬레이션을 수행하는 프로젝트다. 핵심 목표는 다음과 같다.

- ETF별 시장 상관관계, 수급 예측력, 가격 rotation signal을 일자별로 계산한다.
- 계산된 rank panel을 사용해 A0~A5 모델의 다음 거래일 open 리밸런싱을 시뮬레이션한다.
- 백테스트와 파일럿 모두 현금/주식 regime 비중, commission, 정수 ETF 수량 체결을 반영한다.
- 모델별 daily PnL, target weights, 실제 거래 원장, 성과 요약, 그래프를 남긴다.
- LLM 모델은 GitHub Actions에서는 NVIDIA API, 로컬 pilot에서는 Z.AI GLM-4.7-Flash API를 사용할 수 있다.

이 README는 이 프로그램을 처음 보는 사람이 데이터 흐름, 계산 방식, 산출물 의미를 이해할 수 있도록 작성했다.

## Directory Layout

```text
ML6/
  testing/
    01_make_market_corr_rank.py
    02_make_flow_price_rank.py
    03_make_rotation_rank.py
    04_merge_rank_panel.py
    A0_equal_weight.py
    A1_rule_based_rank_allocator.py
    A2a_llm_opaque_rank_allocator.py
    A2b_llm_semantic_rank_allocator.py
    A3_llm_policy_pack_allocator.py
    A4_rule_based_llm_blend.py
    A5_bayesian_winner_loser_allocator.py
    run_full_backtest.py
    plot_trade_history_report.py
    sector_all_merged.csv
    sector_fund_flow.csv
    regime_weights.csv
  pilot/
    kiwoom_fetch_data.py
    pilot_run_all.py
    pilot_run_A0_equal_weight.py
    pilot_run_A1_rule_based.py
    pilot_run_A2_llm.py
    pilot_run_A3_llm_policy.py
    pilot_run_A4_rule_llm_blend.py
    pilot_run_A5_bayesian.py
  .github/workflows/
    etf-rank-backtest-a0-a5.yml
```

`testing`은 전체 기간 백테스트용이고, `pilot`은 실제 운용처럼 특정 시작일 이후 state를 이어가며 다음 open 리밸런싱 목표를 산출하는 용도다.

## Input Data

### `testing/sector_all_merged.csv`

ETF 가격, 거래량, 거래대금, KOSPI 종가가 들어 있는 일자별 파일이다.

주요 컬럼:

- `date`: 거래일
- `ticker`: ETF 종목코드
- `name`: ETF 이름
- `open`, `high`, `low`, `close`: ETF OHLC 가격
- `volume`: 거래량
- `amount`: 거래대금, 백만원 단위로 간주
- `gics_sector`: 섹터 분류
- `kospi_close`: KOSPI 종가

### `testing/sector_fund_flow.csv`

ETF별 투자자 주체 수급 데이터다.

주요 컬럼:

- `date`, `ticker`
- `individual_net_buy`
- `foreign_net_buy`
- `institution_net_buy`
- `finance_invest_net_buy`
- `insurance_net_buy`
- `trust_net_buy`
- `other_finance_net_buy`
- `bank_net_buy`
- `pension_net_buy`
- `private_fund_net_buy`
- `nation_net_buy`
- `other_corp_net_buy`
- `other_foreign_net_buy`
- `총 FF`

수급 컬럼은 천원 단위로 간주한다. 따라서 수급 ratio를 만들 때 가격 파일의 `amount` 백만원 단위와 맞추기 위해 `amount * 1000`을 분모로 쓴다.

### `testing/regime_weights.csv`

현금/주식 비중을 결정하는 weekly regime 파일이다.

주요 컬럼:

- `date`: regime 기준일
- `regime_key`: regime 이름
- `cash_weight`: 현금 비중
- `stock_weight`: ETF 전체 sleeve에 배정할 주식 비중
- `regime_name`

백테스트나 pilot의 특정 실행일에는 그 일자보다 같거나 빠른 가장 최근 regime row를 사용한다. 현재 2026년 pilot은 파일 마지막 row의 `DRAI_RISK_ON`, `cash_weight=0.05`, `stock_weight=0.95`가 적용된다.

## Timing Convention

모든 전략의 기본 타임라인은 같다.

1. `decision_date` close까지의 데이터로 rank와 모델 weight를 계산한다.
2. 계산된 weight는 다음 거래일인 `execution_date` open에서 체결한다고 가정한다.
3. 체결 후 `execution_date` close 기준 NAV와 PnL을 평가한다.
4. 다음 `decision_date` close에서 다시 다음 거래일 open target을 계산한다.

이 구조는 look-ahead를 막기 위한 핵심 규칙이다. 예를 들어 2026-05-04 close 데이터로 계산한 weight는 2026-05-06 open 리밸런싱 예정 비중이 된다.

## Trading Assumptions

- 초기 NAV:
  - 전체 백테스트 기본값: `1,000,000,000 KRW`
  - pilot 기본값: `100,000,000 KRW`
- commission: 매수와 매도 양쪽 모두 one-way `0.00015`
- turnover 기준 commission:
  - `commission = 0.00015 * sum(abs(target_value_open - old_value_open))`
- slippage, 세금, ETF 괴리, 호가 스프레드, 분배금 보정은 현재 scope 밖이다.
- ETF는 소수점 수량 거래가 불가능하므로 실제 체결 수량은 항상 내림 처리한다.
  - `target_qty = floor(target_value / open_price)`
  - 내림 후 남는 금액은 현금으로 유지한다.
- regime이 주식 비중 95%, 현금 비중 5%라면 NAV 전체가 아니라 `0.95 * NAV_after_commission`만 ETF target value 계산에 사용한다.

## Rank Pipeline

rank pipeline은 `01 -> 02 -> 03 -> 04` 순서로 실행된다.

### 01. Market Correlation Rank

파일: `testing/01_make_market_corr_rank.py`

목적:

- ETF가 KOSPI와 얼마나 같이 움직이는지 측정한다.
- 현재 v1 rank 기준에서는 market correlation이 낮을수록 더 좋은 diversification signal로 간주한다.

사용 데이터:

- ETF `close`
- KOSPI `kospi_close`
- tradability 확인용 `open`, `volume`, `amount`

계산 방식:

1. ETF close-to-close log return을 계산한다.
   - `etf_cc_ret = log(close_t / close_{t-1})`
2. KOSPI close-to-close log return을 계산한다.
   - `index_cc_ret = log(index_close_t / index_close_{t-1})`
3. ETF별 rolling correlation을 계산한다.
   - `corr(etf_cc_ret, index_cc_ret)`
4. ETF별 rolling beta도 계산한다.
   - `cov(etf_cc_ret, index_cc_ret) / var(index_cc_ret)`
5. open, close, volume, amount가 정상이고 rolling observation이 충분한 row만 valid로 둔다.
6. market rank는 낮은 correlation이 더 좋은 rank가 되도록 매긴다.
   - rank 1: KOSPI와 최근 동행성이 가장 낮은 ETF

산출물:

- `testing/market_rank.csv`
- 주요 컬럼:
  - `market_corr_active`
  - `market_corr_rolling20`
  - `market_corr_ewma20`
  - `market_beta_rolling20`
  - `market_score`
  - `market_rank`
  - `market_valid`

### 02. Flow Price Rank

파일: `testing/02_make_flow_price_rank.py`

목적:

- 투자자 주체별 수급이 다음 시점 ETF 수익률을 예측하는지 rolling 방식으로 평가한다.
- 예측력이 threshold를 넘는 actor만 사용해 ETF별 flow score를 만든다.

사용 데이터:

- `sector_all_merged.csv`
- `sector_fund_flow.csv`

단위 처리:

- `net_buy`: 천원 단위
- `amount`: 백만원 단위
- 따라서 flow ratio는 다음처럼 계산한다.
  - `flow_ratio = net_buy / (amount * 1000)`
- `amount <= 0`이면 flow ratio는 0으로 둔다. 아주 작은 EPS로 나누지 않는다. 수급이 없는 날을 과장하지 않기 위해서다.

actor 구성:

- 원천 actor:
  - 개인, 외국인, 기관, 금융투자, 보험, 투신, 기타금융, 은행, 연기금, 사모펀드, 국가, 기타법인, 기타외국인
- 파생 actor:
  - `total_net_buy`
  - `총 FF`가 있으면 사용하고, 없으면 `foreign_net_buy + institution_net_buy`로 대체한다.

계산 방식:

1. actor별 flow ratio를 만든다.
2. actor별 rolling z-score를 만든다.
   - rolling mean/std는 `shift(1)`을 적용해 현재 flow 값이 자기 자신의 기준값 계산에 들어가지 않게 한다.
   - rolling std가 0이면 해당 actor z-score contribution은 0이다.
3. actor별 predictive correlation을 계산한다.
   - 설명 변수: actor z-score
   - 예측 대상: 다음 거래일 open-to-close return
   - correlation 값도 `shift(1)`을 적용해 decision date 기준 이미 확정된 과거 label만 사용한다.
4. observation 수가 적은 correlation은 shrink한다.
   - `shrunk_corr = raw_corr * obs / (obs + shrink_k)`
5. actor 선택:
   - `abs(shrunk_corr) >= corr_threshold`인 actor만 선택한다.
6. 방향성:
   - correlation이 양수면 `+zscore`
   - correlation이 음수면 `-zscore`
7. ETF별 raw score:
   - 선택 actor가 있으면 `mean(sign(corr) * actor_zscore)`
   - 선택 actor가 없으면 0
8. 일자별 cross-section에서 score가 높을수록 좋은 flow rank를 부여한다.

산출물:

- `testing/flow_rank.csv`
- `testing/flow_rank_metadata.json`
- `testing/flow_rank_no_selected_actor_report.csv`
- 주요 컬럼:
  - `flow_score_raw`
  - `flow_score`
  - `flow_rank`
  - `flow_valid`
  - `flow_selected_actor_count`
  - `flow_selected_actors`

### 03. Rotation Rank

파일: `testing/03_make_rotation_rank.py`

목적:

- 최근 가격 momentum 또는 sector rotation 강도를 측정한다.

사용 데이터:

- ETF `close`
- tradability 확인용 `open`, `volume`, `amount`

계산 방식:

1. ETF별 close-to-close log return을 계산한다.
   - `ret_1d = log(close_t / close_{t-1})`
2. 같은 날짜의 ETF들끼리 cross-sectional z-score를 계산한다.
3. z-score가 높을수록 강한 rotation signal로 본다.
4. score가 높을수록 좋은 rank를 부여한다.

산출물:

- `testing/rotation_rank.csv`
- 주요 컬럼:
  - `ret_1d`
  - `rotation_score_raw`
  - `rotation_score`
  - `rotation_rank`
  - `rotation_valid`

### 04. Rank Panel Merge

파일: `testing/04_merge_rank_panel.py`

목적:

- market, flow, rotation rank를 하나의 point-in-time panel로 합친다.
- 이후 모든 모델은 이 `rank_panel.csv`만 사용한다.

계산 방식:

1. 가격 panel을 기준으로 market/flow/rotation rank를 `date`, `ticker`로 병합한다.
2. ticker별 `ETF_id`, `asset_id` mapping을 만든다.
3. tradability를 계산한다.
   - open, close, volume, amount가 정상인 ETF만 tradable
4. 모든 rank가 valid인 row에만 `all_rank_valid=True`를 부여한다.
5. `future`, `next_open`, `next_close`, `winner_label` 같은 look-ahead 위험 컬럼이 있으면 실패 처리한다.

산출물:

- `testing/rank_panel.csv`
- `testing/id_mapping.csv`

## Model Pipeline

모든 모델은 `rank_panel.csv`의 `tradable & all_rank_valid` ETF만 대상으로 한다. 모델이 산출하는 weight는 ETF sleeve 내부 비중이다. 실제 포트폴리오 비중은 `stock_weight * sleeve_weight`다.

### A0 Equal Weight

파일: `testing/A0_equal_weight.py`

목적:

- 가장 단순한 baseline이다.
- valid ETF 전체를 동일 비중으로 보유한다.

계산 방식:

1. decision date의 valid ETF를 찾는다.
2. 모든 valid ETF에 동일 sleeve weight를 부여한다.
3. execution date open에서 regime stock weight만큼 매수한다.
4. 정수 수량 내림 후 남은 금액은 현금으로 보유한다.

### A1 Rule-Based Rank Allocator

파일: `testing/A1_rule_based_rank_allocator.py`

목적:

- 세 rank score를 고정 가중 평균해 rule-based allocation을 만든다.

계산 방식:

```text
composite_score =
  0.30 * market_score
+ 0.35 * flow_score
+ 0.35 * rotation_score
```

score가 음수로 내려가면 0으로 clip한다. top-k 설정이 10이면 전체 valid ETF에 score 비례 weight를 준다. score 합이 0이면 equal weight fallback을 쓴다.

### A2 LLM Semantic Allocator

파일:

- `testing/A2a_llm_opaque_rank_allocator.py`
- `testing/A2b_llm_semantic_rank_allocator.py`

pilot 기본 A2 variant:

- `A2b`

목적:

- LLM이 rank와 score를 보고 직접 ETF sleeve weight를 결정한다.

A2a와 A2b 차이:

- A2a: ticker/name/sector를 숨기고 generic signal rank/score만 제공한다.
- A2b: rank 의미를 명시한다.
  - market rank: 낮은 시장 상관관계, diversification/co-movement signal
  - flow rank: 예측력 threshold를 통과한 actor들의 방향 조정 수급 signal
  - rotation rank: 최근 close-to-close 가격 rotation signal

검증:

- LLM은 JSON만 반환해야 한다.
- `target_weights`의 asset_id가 유효해야 한다.
- sleeve weight 합은 1이어야 한다.
- 음수 weight는 허용하지 않는다.
- validation 실패 또는 API 실패 시 A1 rule-based weight로 fallback한다.

### A3 LLM Policy Pack Allocator

파일: `testing/A3_llm_policy_pack_allocator.py`

목적:

- A2b semantic prompt에 policy pack을 추가한다.

policy pack 내용:

- rank 1이 best
- higher score가 better
- rank alignment가 강하면 conviction 증가
- rank conflict가 있으면 concentration 억제
- 불필요한 turnover를 피하되, constraint를 만족해야 함
- reason code를 반환할 수 있음

validation 실패 시 A1 fallback을 쓴다.

### A4 Rule-Based LLM Blend

파일: `testing/A4_rule_based_llm_blend.py`

목적:

- A1 rule-based weight를 기준점으로 제공하고, LLM weight와 혼합한다.

계산 방식:

1. A1 base weight를 먼저 계산한다.
2. LLM에게 rank/score/current weight/base weight/policy pack을 제공한다.
3. LLM output이 valid하면 base weight와 LLM weight를 blend한다.
4. LLM output이 invalid하면 A1 base weight를 그대로 쓴다.

### A5 Bayesian Winner-Loser Allocator

파일: `testing/A5_bayesian_winner_loser_allocator.py`

목적:

- rank state별로 과거 benchmark 대비 승률을 학습해 posterior score를 만든다.

label:

- `ETF next close-to-close return > index next close-to-close return`

중요한 점:

- label은 history fitting에만 사용한다.
- 현재 decision date의 미래 label은 사용하지 않는다.
- `get_bayesian_history()`는 `date < decision_date`만 가져온다.

계산 방식:

1. market/flow/rotation score를 bucket으로 나눈다.
   - top, middle, bottom
2. 세 bucket 조합을 `state_key`로 만든다.
3. 과거 history에서 state_key별 benchmark beat 횟수와 실패 횟수를 센다.
4. Beta prior로 posterior mean을 계산한다.
   - `alpha0=1`, `beta0=1`
5. 관측 수가 적은 state는 global posterior mean으로 shrink한다.
6. 현재 ETF의 state_key에 posterior score를 붙인다.
7. posterior score 비례로 sleeve weight를 만든다.

## Backtest Outputs

각 모델 출력 폴더에는 다음 파일이 생성된다.

### `daily_results.csv` 또는 `daily_pnl.csv`

일자별 portfolio-level 성과 파일이다.

주요 컬럼:

- `decision_date`
- `execution_date`
- `nav_open_pre`
- `nav_open_post`
- `nav_close`
- `daily_return`
- `commission`
- `turnover_value`
- `turnover_ratio`
- `stock_ratio`
- `fallback_used`
- `effective_num_positions`

pilot에서는 파일명이 `daily_pnl.csv`다.

### `weights.csv`

testing 백테스트에서 모델이 산출한 sleeve weight와 실제 portfolio weight를 저장한다.

주요 컬럼:

- `decision_date`
- `execution_date`
- `ticker`
- `ETF_id`
- `sleeve_weight`
- `portfolio_weight`

### `trade_history.csv`

testing과 pilot 모두 동일 schema로 생성되는 거래 원장이다. 이 파일이 모델별 실제 체결 검증의 핵심이다.

주요 컬럼:

- `strategy`
- `decision_date`
- `execution_date`
- `ticker`
- `ETF_id`
- `name`
- `open`
- `close`
- `prev_qty`: 리밸런싱 직전 보유 수량
- `target_qty`: 리밸런싱 후 목표 수량, 정수 수량
- `delta_qty`: 실제 주문 수량
- `side`: BUY, SELL, HOLD
- `prev_value_open`
- `target_value_open`
- `delta_value_open`
- `trade_turnover_value`
- `day_turnover_value`
- `commission_allocated`
- `day_commission`
- `nav_open_pre`
- `nav_open_post`
- `nav_close`
- `daily_return`
- `stock_ratio`
- `cash_weight`
- `target_sleeve_weight`
- `target_portfolio_weight`
- `prev_portfolio_weight_open`
- `actual_portfolio_weight_open`
- `cash_after_rebalance`
- `rebalanced`

이 schema는 testing과 pilot이 동일하다. 따라서 같은 plotting script에 testing CSV와 pilot CSV를 모두 넣을 수 있다.

### `summary.json` 또는 `latest_summary.csv`

성과 요약 파일이다.

주요 지표:

- final NAV
- total return
- CAGR
- annual volatility
- Sharpe
- Sortino
- MDD
- Calmar
- total commission
- commission drag
- total turnover
- average turnover ratio
- fallback rate 또는 fallback 여부

## Plotting

파일: `testing/plot_trade_history_report.py`

목적:

- 하나 이상의 `trade_history.csv`를 입력으로 받아 한 장의 PNG에 다음을 그린다.
  - 모델별 NAV/PnL path
  - 모델별 total return bar
  - 모델별 Sharpe bar
  - 모델별 total turnover / initial NAV bar

사용 예:

```powershell
python .\testing\plot_trade_history_report.py `
  .\pilot\out_20260401_zai_flash_trade_full_json\A0\trade_history.csv `
  .\pilot\out_20260401_zai_flash_trade_full_json\A5\trade_history.csv `
  --out .\pilot\out_20260401_zai_flash_trade_full_json\pilot_A0_A5_report.png
```

CSV 경로를 A0, A5만 넣으면 그래프에는 A0, A5만 나온다. A2~A5를 넣으면 A2~A5만 나온다.

파일 상단의 `TRADE_CSV_PATHS`를 직접 수정해도 된다.

## Full Backtest with Docker

`testing/docker-compose.yml`은 testing 폴더를 `/app`으로 mount하고 `run_full_backtest.py`를 실행한다.

로컬 실행 예:

```powershell
cd .\testing
docker compose up --build
```

주요 환경변수:

- `REBUILD_RANKS=1`: 01~04 rank를 처음부터 재생성
- `RUN_A0_A1_A5=1`: A0, A1, A5 실행
- `RUN_LLM=1`: A2~A4 실행
- `RUN_LLM_PARALLEL=1`: A2~A4 병렬 실행
- `RUN_A6=0`: 현재 GitHub Actions에서는 A6 제외
- `TOP_K_VALUES=10`: 현재 workflow는 10만 실행
- `REGIME_CSV_PATH=/app/regime_weights.csv`

## GitHub Actions

workflow:

- `.github/workflows/etf-rank-backtest-a0-a5.yml`

이 workflow는 수동 실행(`workflow_dispatch`)이다.

동작:

1. repository를 checkout한다.
2. `NVIDIA_API_KEY` GitHub Actions secret이 있는지 확인한다.
3. Docker compose로 rank 재생성부터 A0~A5 전체 백테스트를 실행한다.
4. 결과를 artifact로 업로드한다.

필요한 secret:

- `NVIDIA_API_KEY`

`zai_llm_config.json` 전체를 secret으로 만들 필요는 없다. API key만 secret이면 된다.

주의:

- `.gitignore`에 `testing/out_*/`가 있어도 GitHub Actions artifact 업로드에는 문제가 없다.
- `.gitignore`는 Git commit 대상에서 제외할 뿐이다.
- workflow가 run 중 생성한 `testing/out_actions_flow_v2_a0_a5_k10/` 파일은 `actions/upload-artifact`가 직접 경로를 읽어 업로드한다.

## Pilot

파일: `pilot/pilot_run_all.py`

pilot은 실제 운용에 가까운 stateful 실행을 목표로 한다.

핵심 구조:

1. 최초 실행일 open에 초기 target weight를 체결한다.
2. 매 거래일 close 기준으로 다음 거래일 open target weight를 계산한다.
3. 다음 거래일 open에 이전 close에서 계산한 target을 체결한다.
4. 기존 `state.json`과 `daily_pnl.csv`가 있으면 이미 처리한 날짜는 다시 계산하지 않는다.
5. 새 as-of date가 들어오면 마지막 처리일 이후 날짜만 append한다.

예시:

```powershell
$env:LLM_CONFIG_PATH = "C:\Users\박제형\Desktop\ML6\testing\zai_glm_4_7_flash_local_config.json"

python .\pilot\pilot_run_all.py `
  --skip-fetch `
  --start-date 2026-04-01 `
  --as-of-date 2026-05-04 `
  --next-open-date 2026-05-06 `
  --top-k 10 `
  --initial-nav 100000000 `
  --strategies A0,A1,A2,A3,A4,A5 `
  --out-dir .\pilot\out_20260401_zai_flash_trade_full_json
```

### Local Z.AI Flash Config

로컬 pilot용 파일:

- `testing/zai_glm_4_7_flash_local_config.json`

이 파일은 `.gitignore`에 들어 있으므로 GitHub에 올리지 않는다.

중요 설정:

- `model`: `glm-4.7-flash`
- `response_format`: JSON object
- `thinking.type`: `disabled`

`thinking`을 끄지 않으면 Z.AI flash가 `content` 대신 `reasoning_content`만 길게 반환할 수 있고, 이 경우 모델 output이 JSON으로 파싱되지 않아 A1 fallback이 발생한다.

## Pilot Verification Result

검증 실행 폴더:

- `pilot/out_20260401_zai_flash_trade_full_json`

조건:

- 기간: 2026-04-01 open 시작, 2026-05-04 close까지 평가
- 다음 target: 2026-05-06 open
- 초기 NAV: 100,000,000 KRW
- regime: `DRAI_RISK_ON`
- cash/stock: 5% / 95%
- LLM: local Z.AI `glm-4.7-flash`
- A2~A4 fallback: false

일자/거래 파일 검증:

| Strategy | Daily Rows | Rebalanced Days | Nonzero Turnover Days | Trade Rows | Nonzero Trade Rows |
|---|---:|---:|---:|---:|---:|
| A0 | 23 | 23 | 23 | 230 | 220 |
| A1 | 23 | 23 | 23 | 230 | 228 |
| A2 | 23 | 23 | 23 | 154 | 151 |
| A3 | 23 | 23 | 23 | 135 | 131 |
| A4 | 23 | 23 | 23 | 230 | 229 |
| A5 | 23 | 23 | 23 | 230 | 226 |

성과 요약:

| Strategy | NAV Close | Total Return | Sharpe | Total Commission | Total Turnover | Avg Turnover Ratio |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 114,636,791.84 | 14.64% | 7.678 | 18,012.16 | 120,081,084 | 0.0514 |
| A1 | 113,778,436.83 | 13.78% | 7.241 | 128,437.17 | 856,247,795 | 0.3512 |
| A2 | 112,777,015.67 | 12.78% | 7.554 | 169,036.33 | 1,126,908,843 | 0.4647 |
| A3 | 105,263,086.28 | 5.26% | 2.635 | 219,897.72 | 1,465,984,791 | 0.6005 |
| A4 | 113,487,599.25 | 13.49% | 7.477 | 178,791.75 | 1,191,944,970 | 0.4838 |
| A5 | 114,821,024.59 | 14.82% | 7.860 | 43,799.41 | 291,996,067 | 0.1206 |

생성된 그래프:

- `pilot/out_20260401_zai_flash_trade_full_json/pilot_A0_A5_full_report.png`

## Continuity Verification

연속 운용 구조도 확인했다.

검증 방식:

1. 같은 out 폴더에 2026-04-01부터 2026-04-30까지 먼저 실행했다.
2. 이후 같은 out 폴더에 2026-05-04 as-of를 이어 실행했다.
3. 기존 2026-04-01~2026-04-30 daily row 값이 변하지 않는지 확인했다.
4. 2026-05-04 row만 추가되는지 확인했다.
5. `state.json`의 `pending_execution_date`가 2026-05-06으로 갱신되는지 확인했다.

결과:

- A0: 기존 22개 row 유지, 2026-05-04 1개 row 추가, prior values unchanged
- A5: 기존 22개 row 유지, 2026-05-04 1개 row 추가, prior values unchanged
- `trade_history.csv`도 2026-05-04 execution row까지 append됨
- `state.json`은 다음 open인 2026-05-06 target을 보유

따라서 pilot은 전체 과거 PnL을 매번 재작성하는 방식이 아니라, 이미 처리한 날짜를 보존하고 신규 날짜를 이어 붙이는 방식으로 작동한다. 단, rank 계산 파일은 새 데이터가 들어오면 rolling rank 계산을 위해 panel을 다시 생성할 수 있다. 그러나 실제 체결 state, 보유 수량, daily PnL은 기존 결과에서 이어진다.

## Kiwoom Data Fetch

파일: `pilot/kiwoom_fetch_data.py`

목적:

- Kiwoom API에서 최근 ETF OHLCV, ETF 수급, KOSPI close를 받아 pilot data CSV에 병합한다.

환경변수는 `pilot/.env`에 둔다. 이 파일은 `.gitignore`에 포함되어 GitHub에 올리지 않는다.

주요 변수:

- `KIWOOM_APP_KEY`
- `KIWOOM_SECRET_KEY`
- `KIWOOM_ACCOUNT_NO`
- `KIWOOM_USE_MOCK`
- `KIWOOM_BASE_URL`
- `KIWOOM_TOKEN_URL`
- `KIWOOM_REQUEST_SLEEP_SECONDS`
- `KIWOOM_MAX_RETRIES`

현재 pilot 검증은 이미 확보된 `pilot/data`와 `testing` fallback CSV를 사용했다.

## Safety Notes

- API key 파일은 GitHub에 올리지 않는다.
  - `testing/zai_llm_config.json`
  - `testing/zai_glm_4_7_flash_local_config.json`
  - `pilot/.env`
- LLM output이 invalid하면 A1 fallback이 발생한다.
- fallback 여부는 `state.json`, `latest_summary.csv`, `decision_log.jsonl`에서 확인한다.
- 백테스트 결과는 survivorship bias를 포함할 수 있다. universe는 CSV에 존재하는 ETF로 제한된다.
- 현재 market rank는 낮은 KOSPI correlation을 선호한다. bull/bear/neutral regime별로 high/low correlation 선호를 바꾸는 방식은 아직 rank 계산에 반영하지 않았다.

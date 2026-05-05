# ETF Rank Pipeline Calculation Details

이 문서는 `testing` 폴더의 `01~04`, `A0~A6` 파일이 어떤 데이터를 사용하고, 어떤 계산을 하며, 어떤 결과물을 남기는지 정리한다. 아래 성과표는 현재 디스크에 저장된 완료 `summary.json` 기준이다. 단, flow rank 로직은 이후 `selected actor threshold` 방식으로 수정되었으므로 기존 저장 성과는 재산출 전 결과다.

## 공통 데이터

### 입력 CSV

- `sector_all_merged.csv`
  - 일자별 ETF 가격/거래 데이터.
  - 주요 컬럼: `date`, `ticker`, `name`, `open`, `high`, `low`, `close`, `volume`, `amount`, `gics_sector`, `kospi_close`.
  - `amount`는 백만원 단위로 간주한다.
  - `kospi_close`는 `rank_panel.csv` 병합 시 `index_close`로 표준화된다.

- `sector_fund_flow.csv`
  - 일자별 ETF 투자자 주체별 순매수 데이터.
  - 주요 컬럼: `individual_net_buy`, `foreign_net_buy`, `institution_net_buy`, 세부 기관별 net buy 컬럼, `총 FF`, `ticker`.
  - net buy 계열은 천원 단위로 간주한다.

### 공통 백테스트 가정

- 의사결정 시점: `decision_date` 장마감 후.
- 체결 시점: 다음 거래일 `execution_date` open.
- 평가 시점: `execution_date` close.
- 초기 NAV: 1,000,000,000 KRW.
- commission: one-way `0.00015`.
- 거래비용 계산: 리밸런싱 전 open 평가금액과 target value 차이의 절대값 합에 commission rate를 곱한다.
- 저장된 `rank_panel.csv`에는 `next_*`, `future_*`, `winner_label` 컬럼을 허용하지 않는다.

## 01_make_market_corr_rank.py

### 데이터 사용

- 입력: `sector_all_merged.csv`.
- 사용 컬럼: ETF `close`, `open`, `volume`, `amount`, `kospi_close`.
- `kospi_close`는 내부에서 `index_close`로 표준화된다.

### 계산 방식

1. ETF close-to-close log return 계산:
   - `etf_cc_ret = log(close_t / close_{t-1})`
2. KOSPI close-to-close log return 계산:
   - `index_cc_ret = log(index_close_t / index_close_{t-1})`
3. ETF별 rolling market correlation 계산:
   - `corr(etf_cc_ret, index_cc_ret)`
4. ETF별 rolling beta 계산:
   - `cov(etf_cc_ret, index_cc_ret) / var(index_cc_ret)`
5. 유효 조건:
   - open/close/volume/amount 양수
   - 충분한 rolling observation
   - correlation finite
6. rank:
   - 낮은 market correlation이 더 좋은 rank.
   - rank 1은 시장과 가장 덜 같이 움직이는 ETF.

### 결과물

- `market_rank.csv`
- 주요 컬럼: `market_corr_active`, `market_corr_rolling20`, `market_corr_ewma20`, `market_beta_rolling20`, `market_score`, `market_rank`, `market_valid`.

## 02_make_flow_price_rank.py

### 데이터 사용

- 입력:
  - `sector_all_merged.csv`
  - `sector_fund_flow.csv`
- 가격 데이터의 `amount`와 수급 데이터의 net buy 단위를 맞춰 flow ratio를 계산한다.

### 계산 방식

현재 수정된 방식은 actor selection 기반이다.

1. `total_net_buy` 생성:
   - `총 FF` 컬럼이 있으면 이를 사용.
   - 없으면 `foreign_net_buy + institution_net_buy`를 사용.
2. actor별 flow ratio:
   - `amount > 0`이면 `net_buy / (amount * 1000)`
   - `amount <= 0`이면 `0`
3. actor별 rolling z-score:
   - rolling mean/std는 `shift(1)`을 적용해 현재일 flow ratio가 자기 자신의 기준값에 들어가지 않도록 한다.
   - rolling std가 0이면 z-score는 0 contribution.
4. actor별 predictive correlation:
   - actor z-score와 다음 거래일 open-to-close return의 rolling correlation을 계산한다.
   - correlation에는 `shift(1)`을 적용해 decision date 기준 이미 확정된 과거 label까지만 사용한다.
5. shrink:
   - observation 수가 적은 actor correlation은 `obs / (obs + shrink_k)`로 축소한다.
6. actor 선택:
   - `abs(shrunk_predictive_corr) >= corr_threshold`인 actor만 선택한다.
   - 방향성은 correlation 부호로 결정한다.
7. raw score:
   - 선택 actor가 있으면 `mean(sign(corr) * actor_flow_z)`.
   - 선택 actor가 없으면 raw score는 0.
8. rank:
   - `flow_score_raw`가 클수록 rank가 좋다.

### 미선택 actor 처리

- 선택 actor가 없는 row는 `flow_rank_no_selected_actor_report.csv`에 기록한다.
- 사용자가 휴리스틱 override를 정하려면 이 report의 `date`, `ticker`, `name`, `flow_obs`를 기준으로 판단한다.

### 결과물

- `flow_rank.csv`
- `flow_rank_no_selected_actor_report.csv`
- `flow_rank_metadata.json`
- 주요 컬럼: `flow_score_raw`, `flow_score`, `flow_rank`, `flow_valid`, `flow_selected_actor_count`, `flow_selected_actors`.

## 03_make_rotation_rank.py

### 데이터 사용

- 입력: `sector_all_merged.csv`.
- 사용 컬럼: ETF `close`, `open`, `volume`, `amount`.

### 계산 방식

1. ETF별 close-to-close log return:
   - `ret_1d = log(close_t / close_{t-1})`
2. 일자별 cross-sectional z-score:
   - 같은 일자의 ETF들끼리 `ret_1d`를 표준화한다.
3. rank:
   - `rotation_score_raw`가 높을수록 rank가 좋다.
   - rank 1은 최근 가격 rotation이 가장 강한 ETF.

### 결과물

- `rotation_rank.csv`
- 주요 컬럼: `ret_1d`, `rotation_score_raw`, `rotation_score`, `rotation_rank`, `rotation_valid`.

## 04_merge_rank_panel.py

### 데이터 사용

- 입력:
  - `sector_all_merged.csv`
  - `market_rank.csv`
  - `flow_rank.csv`
  - `rotation_rank.csv`

### 계산 방식

1. 가격 panel을 표준화한다.
2. ticker별 `ETF_id`, `asset_id` mapping을 생성한다.
3. `date`, `ticker` 기준으로 market/flow/rotation rank를 one-to-one merge한다.
4. `tradable` flag:
   - open/close/volume/amount 양수.
5. `all_rank_valid`:
   - `market_valid & flow_valid & rotation_valid`.
6. forbidden future/label 컬럼이 있으면 실패시킨다.

### 결과물

- `rank_panel.csv`
- `id_mapping.csv`

## A0_equal_weight.py

### 데이터 사용

- 입력: `rank_panel.csv`.
- 매 decision date마다 `tradable & all_rank_valid`인 ETF만 사용한다.

### 계산 방식

- valid ETF에 동일 sleeve weight를 부여한다.
- 다음 거래일 open에서 target weight로 리밸런싱한다.
- 다음 거래일 close 기준 NAV를 기록한다.

### 결과물

- `daily_results.csv`
- `weights.csv`
- `summary.json`
- `decision_log.jsonl`

## A1_rule_based_rank_allocator.py

### 데이터 사용

- 입력: `rank_panel.csv`.
- 사용 컬럼: `market_score`, `flow_score`, `rotation_score`.

### 계산 방식

- composite score:
  - `0.30 * market_score + 0.35 * flow_score + 0.35 * rotation_score`
- score 비례 weight를 만든다.
- 다음 거래일 open에서 체결한다.

### 결과물

- A0와 동일한 백테스트 결과 파일 세트.

## A2a_llm_opaque_rank_allocator.py

### 데이터 사용

- 입력: `rank_panel.csv`.
- 각 decision date의 valid ETF만 LLM prompt에 제공한다.
- ticker/name/sector 의미는 숨기고 `asset_id`, signal rank/score만 제공한다.

### 계산 방식

- LLM은 세 개의 generic signal만 보고 sleeve weight JSON을 반환한다.
- JSON validation 실패 또는 API 실패 시 A1 base weight로 fallback한다.
- checkpoint를 매 decision date마다 저장하므로 중단 후 재실행 가능하다.

### 결과물

- `daily_results.csv`
- `weights.csv`
- `decision_log.jsonl`
- `checkpoint_state.json`
- `summary.json` 완료 시 생성

## A2b_llm_semantic_rank_allocator.py

### 데이터 사용

- 입력: `rank_panel.csv`.
- A2a와 달리 signal 의미를 LLM에 설명한다.

### 계산 방식

- market rank:
  - 낮은 broad-market rolling correlation을 선호하는 diversification/co-movement signal.
- flow rank:
  - 과거 predictive correlation threshold를 넘은 investor-flow actor의 z-score를 방향 조정 후 equal-weight한 signal.
- rotation rank:
  - 최근 close-to-close 가격 rotation signal.
- LLM은 외부 지식 없이 제공된 rank/score/current weight만 사용한다.

### 결과물

- A2a와 동일한 결과 파일 세트.

## A3_llm_policy_pack_allocator.py

### 데이터 사용

- 입력: `rank_panel.csv`.
- A2b의 semantic packet에 policy pack을 추가한다.

### 계산 방식

- policy pack은 rank 충돌 처리, concentration 억제, 불필요한 turnover 회피 원칙을 제공한다.
- LLM은 weight와 함께 reason code를 반환해야 한다.
- validation 실패 시 A1 base weight로 fallback한다.

### 결과물

- A2a와 동일한 결과 파일 세트.

## A4_rule_based_llm_blend.py

### 데이터 사용

- 입력: `rank_panel.csv`.
- A1 base weight와 A3 스타일 LLM weight를 함께 사용한다.

### 계산 방식

- 먼저 A1 base weight를 계산한다.
- LLM에는 base weight를 reference로 제공한다.
- LLM output이 유효하면 rule-based weight와 LLM weight를 blend한다.
- LLM output이 실패하면 A1 base weight를 그대로 사용한다.

### 결과물

- A2a와 동일한 결과 파일 세트.

## A5_bayesian_winner_loser_allocator.py

### 데이터 사용

- 입력: `rank_panel.csv`.
- 사용 컬럼: `market_score`, `flow_score`, `rotation_score`, ETF close, index close.

### 계산 방식

1. 각 score를 bucket화한다:
   - top / middle / bottom / missing.
2. state key 생성:
   - market bucket, flow bucket, rotation bucket 조합.
3. label:
   - ETF의 다음 close-to-close return이 index의 다음 close-to-close return보다 큰지 여부.
4. posterior:
   - decision date보다 과거인 label만 사용한다.
   - state별 win/loss를 Beta-Bernoulli posterior로 변환한다.
   - state sample이 적으면 global posterior로 shrink한다.
5. current day weight:
   - 현재 ETF의 state key에 해당하는 posterior score를 부여한다.
   - posterior score 비례 weight를 만든다.

### 주의사항

- 이 전략은 rank의 사전 의미를 직접 따르는 allocator가 아니라, rank bucket 조합의 과거 benchmark beat 확률을 학습하는 meta-model이다.
- `missing` state와 invalid row의 posterior 포함 여부는 성과 해석에 영향을 줄 수 있다.

### 결과물

- `daily_results.csv`
- `weights.csv`
- `summary.json`
- `posterior_states.csv`

## A6_bayesian_online_expert_allocator.py

### 데이터 사용

- 입력:
  - `rank_panel.csv`
  - A0~A5 또는 A0~A4 expert return stream

### 계산 방식

- expert별 gross return:
  - `r_t,k = 1 + R_t,k`
- portfolio mixture loss:
  - `loss_t(w) = -log(w · r_t)`
- D-ONS 방식으로 expert mixture weight를 simplex 위에서 업데이트한다.
- index return은 update benchmark가 아니라 reporting benchmark로 사용한다.

### 결과물

- `expert_probs.csv`
- `expert_gross_returns.csv`
- `a6_log_wealth.csv`
- `a6_index_relative_log_wealth.csv`
- `daily_results.csv`
- `weights.csv`
- `summary.json`

## 저장된 백테스트 결과 요약

아래 표는 현재 완료된 저장 결과 중 strategy family별 대표 성과다. A2~A4는 현재 NVIDIA LLM 백테스트가 진행 중이라 완료 summary가 아직 없다.

| Strategy | Final NAV | CAGR | Sharpe | Sortino | MDD | Calmar | Commission Drag | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 Equal Weight | 1,845,189,931 | 24.36% | 1.159 | 1.373 | -15.54% | 1.567 | 1.56% | 0.00% |
| A1 Rule Based | 1,743,708,238 | 21.88% | 1.075 | 1.273 | -16.66% | 1.314 | 5.11% | 0.00% |
| A5 Bayesian Winner/Loser | 1,958,069,167 | 27.02% | 1.138 | 1.515 | -19.85% | 1.361 | 16.86% | 0.00% |
| A6 Core Experts | 1,811,144,418 | 23.54% | 1.137 | 1.356 | -16.20% | 1.454 | 5.07% | 0.00% |
| A6 Full Experts | 1,793,401,783 | 23.11% | 1.121 | 1.330 | -15.76% | 1.466 | 4.37% | 0.00% |

### 진행 중 LLM 백테스트 상태

| Strategy | Completed Decisions | Complete |
|---|---:|---|
| A2a Opaque LLM | 548 | false |
| A2b Semantic LLM | 527 | false |
| A3 Policy LLM | 532 | false |
| A4 Rule/LLM Blend | 488 | false |

## Regime Cash/Stock Overlay 계산 방식

이 부분은 core backtest 파일을 아직 수정하지 않고, 저장된 일별 ETF sleeve weight를 기반으로 post-processing할 때 적용하는 방식이다.

### Regime 데이터

- 입력: `ml8_risk_regime_weights_2024_2025_weekly.csv`
- 주요 컬럼:
  - `date`
  - `regime_key`
  - `cash_weight`
  - `stock_weight`
  - `regime_name`

### 일자 매핑

각 execution date에 대해:

1. regime CSV에서 `regime_date <= execution_date`인 row만 남긴다.
2. 그중 가장 최근 row를 선택한다.
3. 2026년 이후 execution date는 CSV 마지막 row의 risk-on regime을 계속 사용한다.

### weight 변환

기존 전략의 ETF weight는 ETF sleeve 내부 비중이다. 이를 portfolio weight로 바꿀 때:

- `portfolio_weight_i,t = stock_weight_t * sleeve_weight_i,t`
- `cash_weight_t = 1 - stock_weight_t`

예를 들어 risk-on이 `stock_weight=0.95`, `cash_weight=0.05`이면, 전략이 ETF A에 sleeve 20%를 줬을 때 실제 portfolio weight는 19%다.

### PnL 재계산

저장된 `weights.csv`와 `rank_panel.csv`의 execution date open/close를 사용해 다시 계산한다.

1. 전일 close NAV와 보유 수량으로 execution date open의 기존 포지션 가치를 평가한다.
2. regime stock weight를 반영한 target portfolio value를 계산한다.
3. 변경 금액 절대값 합에 commission을 적용한다.
4. target shares를 execution date open에 체결했다고 가정한다.
5. execution date close로 NAV를 평가한다.
6. cash는 무수익으로 둔다.

### 평가 지표

재계산된 daily NAV로 CAGR, Sharpe, Sortino, MDD, Calmar, turnover, total commission, commission drag를 다시 산출한다.

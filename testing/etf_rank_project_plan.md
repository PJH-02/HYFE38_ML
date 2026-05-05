# ETF Rank 기반 포트폴리오 실험 전체 개발 계획 v2

## 0. 프로젝트 목적

본 프로젝트는 섹터 ETF 포트폴리오를 다음 세 가지 rank signal만으로 구성했을 때, rule-based, LLM-based, Bayesian ML, online expert 방식이 각각 어떤 성과와 안정성을 보이는지 비교한다.

사용 가능한 alpha 입력은 세 가지로 제한한다.

| Signal | 의미 | 최종 rank convention |
|---|---|---|
| `market_rank` | ETF와 generic `index`의 rolling correlation 기반 rank | correlation이 낮을수록 rank 상위 |
| `flow_rank` | 주체별 수급의 과거 예측 부호와 당일 수급 증감 기반 rank | flow score가 높을수록 rank 상위 |
| `rotation_rank` | 일별 수익률 기반 순환매 rank | 당일 상대 수익률이 강할수록 rank 상위 |

모든 rank 파일은 다음 규칙을 반드시 만족한다.

```text
rank 1 = 해당 날짜에 가장 선호되는 ETF
rank N = 해당 날짜에 가장 비선호되는 ETF
```

A0~A6 포트폴리오 실험군은 rank 방향을 재해석하지 않는다. Rank 생성 단계에서 이미 방향성이 반영되어 있어야 한다.

---

## 1. 주요 확정 사항

| 항목 | 최종 결정 |
|---|---|
| Regime 데이터 | v1 실험에서 사용하지 않음 |
| 현금/주식 비중 | v1에서는 `stock_ratio = 1.0` 고정. 별도 현금/주식 비중 결정 모듈은 실험 범위 밖 |
| Market corr rank | rolling correlation 1순위, EWMA correlation 2순위 보조 분석 |
| Market rank 방향 | ETF-index correlation이 낮을수록 좋음 |
| Flow rank | 주체별 flow가 미래 수익률에 갖는 양/음 예측력을 rolling으로 추정 후, 오늘 flow 증감에 부호 반영 |
| Rotation rank | close-to-close 일별 수익률 기반 cross-sectional rank |
| 체결 방식 | `t`일 장마감 rank 계산 → `t+1`일 시가 리밸런싱 |
| PnL 처리 | 기존 보유분 overnight PnL + 신규 포지션 open-to-close PnL 모두 반영 |
| 수익률 컬럼 | `rank_panel.csv`에는 future return, next open/close, label 저장 금지 |
| 초기 NAV | `1,000,000,000 KRW` |
| 수수료 | 매수/매도 모두 one-way commission `0.015% = 0.00015` |
| ETF sleeve weight | 항상 `sum = 1`; 잔여 현금 없음 |
| ETF weight cap | 적용하지 않음 |
| `top_k` grid | A1~A6는 `top_k ∈ {3, 5, 7, 10}` 실행. A0는 `top_k`와 무관하게 1회 실행 |
| `top_k=10` 의미 | hard filter를 적용하지 않는다는 뜻. Score-proportional allocation에서 score가 0인 ETF는 weight 0 가능 |
| LLM 입력 | ETF명, ticker, 섹터명, 실제 날짜, `KOSPI` 명칭 노출 금지 |
| LLM용 명칭 | 내부 mapping은 `ETF_id`, LLM prompt에는 더 중립적인 `asset_id` 사용 |
| A4 lambda | 최적화하지 않고 `λ = 0.5` 고정 |
| A5 | Regime 없는 Bayesian benchmark-beat winner/loser posterior |
| A5 winner 정의 | `ETF next close-to-close return > index next close-to-close return` |
| A6 | D-ONS 기반 OBS expert allocator. Core와 full/meta-validation 모드 구분 |
| A6 비용 처리 | expert return은 net-of-commission이면 D-ONS에서 추가 비용 차감 금지. 실제 A6 blended portfolio 리밸런싱 비용은 별도 1회 반영 |
| Index benchmark | close-to-close return만 리포팅용으로 사용. A6 update에는 사용하지 않음 |

---

## 2. LLM 익명화 정책

LLM 실험군에서는 ETF 이름, ticker, 섹터명, 실제 날짜, 대상 시장이 KOSPI라는 사실을 숨긴다.

이 정책은 타당하다. 실험 목적은 LLM이 한국 ETF, 특정 섹터, 특정 시장 국면에 대한 사전학습 지식을 활용하는지 보는 것이 아니라, 동일한 rank signal을 어떻게 조합하는지 보는 것이다. ETF명, ticker, KOSPI, 한국 시장, 섹터명이 노출되면 모델이 사전학습에서 접한 시장 국면이나 특정 ETF/섹터에 대한 prior를 사용할 수 있다.

따라서 LLM prompt 생성 직전에 다음 mapping을 적용한다.

| 실제 데이터 | 내부 관리 | LLM 입력 |
|---|---|---|
| `ticker` | `ticker` 유지 | 사용하지 않음 |
| 익명 ETF 식별자 | `ETF_id` | `asset_id` |
| `name` | 저장 가능 | 사용하지 않음 |
| `gics_sector` | 내부 분석 가능 | 사용하지 않음 |
| `kospi_close` | `index_close` | 직접 제공하지 않음 |
| KOSPI | `index` | `index`라는 generic 용어만 필요 시 사용 |
| 실제 날짜 | `date` 유지 | `decision_step` |

중요 규칙:

```text
rank_panel.csv 내부에서는 ticker를 절대 ETF_id로 덮어쓰지 않는다.
ticker는 merge, 백테스트, 수량 계산, weights 복원에 필요한 내부 key다.
ETF_id는 별도 컬럼으로 생성한다.
LLM 입력은 allowlist 방식으로 asset_id, rank, rank_score, current_sleeve_weight만 선택한다.
```

Mapping 파일은 별도로 저장한다.

```text
id_mapping.csv
    ticker
    ETF_id
    asset_id
```

주의: 익명화가 모든 편향을 완전히 제거하지는 않는다. 가격/수익률 패턴 자체로 시장을 역추론할 가능성은 남아 있다. 그러나 명시적 이름과 시장명으로부터 발생하는 사전지식 오염은 줄일 수 있다.

---

## 3. 전체 파일 구조

```text
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
A6_bayesian_online_expert_allocator.py
```

데이터 흐름은 다음과 같다.

```text
sector_all_merged.csv
sector_fund_flow.csv
        |
        |-- 01_make_market_corr_rank.py  --> market_rank.csv
        |-- 02_make_flow_price_rank.py   --> flow_rank.csv
        |-- 03_make_rotation_rank.py     --> rotation_rank.csv
        |
        |-- 04_merge_rank_panel.py       --> rank_panel.csv
                                                |
                                                |-- A0
                                                |-- A1
                                                |-- A2a
                                                |-- A2b
                                                |-- A3
                                                |-- A4
                                                |-- A5
                                                |-- A6
```

---

## 4. 원천 데이터 스키마

### 4.1 가격 데이터: `sector_all_merged.csv`

필수 컬럼:

```text
date
ticker
name
open
high
low
close
volume
amount
gics_sector
kospi_close
```

용도:

| 컬럼 | 용도 |
|---|---|
| `date` | 거래일 |
| `ticker` | 내부 ETF 식별자. LLM에는 노출하지 않음 |
| `name` | ETF명. LLM에는 노출하지 않음 |
| `open` | ETF 시가. 백테스트 체결 가격 |
| `high` | v1에서는 사용하지 않음. 확장용 |
| `low` | v1에서는 사용하지 않음. 확장용 |
| `close` | ETF 종가. rank 계산 및 장마감 평가 가격 |
| `volume` | 거래량. tradable flag 및 수급 정규화 검증에 사용 |
| `amount` | 거래대금. 백만원 단위로 가정 |
| `gics_sector` | 내부 분석용. LLM에는 노출하지 않음 |
| `kospi_close` | generic `index` 종가로 내부 rename. LLM에는 노출하지 않음 |

### 4.2 수급 데이터: `sector_fund_flow.csv`

주요 컬럼:

```text
date
거래량
individual_net_buy
foreign_net_buy
institution_net_buy
finance_invest_net_buy
insurance_net_buy
trust_net_buy
other_finance_net_buy
bank_net_buy
pension_net_buy
private_fund_net_buy
nation_net_buy
other_corp_net_buy
other_foreign_net_buy
총 FF
flow_source
ticker
name
gics_sector
```

기본 actor list:

```python
FLOW_ACTOR_COLUMNS = [
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
```

`총 FF`는 composite 성격이므로 기본 actor list에는 포함하지 않는다. 별도 보조 분석용 컬럼으로만 유지한다.

---

## 5. 공통 데이터 원칙

### 5.1 Point-in-time 규칙

모든 rank는 `t`일 장마감 이후 계산 가능한 정보만 사용한다.

```text
rank[t]는 t일까지의 가격, 수급, index 정보만 사용한다.
rank[t]는 t+1일 시가 체결에 사용된다.
```

금지:

```text
rank[t] 계산에 t+1 open 사용 금지
rank[t] 계산에 t+1 close 사용 금지
rank[t] 계산에 next-day return 사용 금지
LLM 입력에 next return, future label, next open/close 포함 금지
```

### 5.2 Universe 규칙

날짜별 valid universe는 동적으로 결정한다.

```text
valid ETF 조건:
    open > 0
    close > 0
    volume > 0
    amount > 0
    해당 rank 계산에 필요한 최소 관측치 충족
```

상장 전 ETF는 해당 날짜 universe에 포함하지 않는다. 상장 후에도 rolling window의 최소 관측치가 부족하면 해당 rank는 최소 관측치 충족 전까지 invalid 처리한다.

### 5.3 Survivorship bias 처리

첨부 데이터 universe 내부에서는 다음 방식으로 survivorship bias를 줄인다.

```text
1. 상장 전 ETF forward-fill 금지
2. 특정 ETF가 존재하지 않는 날짜에는 universe에서 제외
3. 날짜별 valid universe 사용
4. A0~A6는 동일 rank_panel.csv와 동일 valid universe 사용
```

다만 원천 데이터가 생존 ETF 10개만 포함한다면, 폐지되었거나 누락된 ETF에 대한 survivorship bias는 완전히 제거할 수 없다. 이 한계는 결과 해석에 명시한다.

---

# 6. `01_make_market_corr_rank.py`

## 6.1 목적

ETF와 generic `index`의 return correlation을 계산하고, correlation이 낮은 ETF를 상위 rank로 둔다. 기본 corr 방식은 rolling이다. EWMA는 보조 계산 및 robustness check용으로만 사용한다.

## 6.2 사용 데이터

입력 데이터는 `sector_all_merged.csv`이다.

필수 컬럼:

```text
date
ticker
name
open
close
volume
amount
gics_sector
kospi_close
```

내부적으로 `kospi_close`는 `index_close`로 rename하여 사용한다.

## 6.3 계산 로직

ETF close-to-close log return:

\[
r_{i,t}=\log\frac{Close_{i,t}}{Close_{i,t-1}}
\]

Index close-to-close log return:

\[
m_t=\log\frac{IndexClose_t}{IndexClose_{t-1}}
\]

Rolling correlation:

\[
\rho^{roll}_{i,t}=Corr(r_{i,t-w+1:t},m_{t-w+1:t})
\]

기본값:

```text
rolling lookback = 20 trading days
min_obs = 20
```

Rolling beta는 보조 지표로 계산한다.

\[
\beta_{i,t}=\frac{Cov(r_i,m)}{Var(m)}
\]

Rank 방향:

```text
market_corr_active 낮을수록 market_rank 상위
```

즉:

\[
MarketRank_{i,t}=RankAscending(\rho^{roll}_{i,t})
\]

## 6.4 출력값: `market_rank.csv`

```text
date
ticker
name
gics_sector
market_corr_active
market_corr_rolling20
market_corr_ewma20
market_beta_rolling20
market_score
market_rank
market_valid
market_corr_mode
market_lookback
market_obs
```

`market_score`:

\[
market\_score_{i,t}=1-\frac{market\_rank_{i,t}-1}{N_t-1}
\]

단, \(N_t=1\)이면 score는 1로 둔다.

## 6.5 생성하고 사용할 함수

### `load_price_data(path)`

```text
CSV 로드
date datetime 변환
ticker string 변환
정렬
kospi_close를 index_close로 내부 rename
```

### `validate_price_schema(df)`

```text
필수 컬럼 존재 여부
date,ticker 중복 여부
open,close,index_close 양수 여부
volume,amount 양수 여부
```

### `compute_close_to_close_returns(df)`

생성 컬럼:

```text
etf_cc_ret
index_cc_ret
```

### `compute_rolling_corr_beta(df, lookback, min_obs)`

ETF별 rolling window에서 다음을 계산한다.

```text
market_corr_rolling20
market_beta_rolling20
market_obs
```

중요 규칙:

```text
date t의 rolling corr는 t일까지의 return만 사용한다.
```

### `compute_ewma_corr(df, halflife)`

보조 지표:

```text
market_corr_ewma20
```

### `select_active_market_corr(df, corr_mode)`

```text
corr_mode = rolling:
    market_corr_active = market_corr_rolling20

corr_mode = ewma:
    market_corr_active = market_corr_ewma20
```

### `make_market_rank(df)`

```text
각 date별 market_valid ETF 선택
market_corr_active 오름차순 정렬
market_rank 생성
market_score 생성
```

### `validate_no_lookahead_market(df)`

```text
market_corr[t]가 t 이후 가격을 사용하지 않았는지 확인
초기 lookback 미달 구간 market_valid=False 확인
```

### `save_market_rank(df, out_path)`

출력 저장.

## 6.6 처리 순서

```text
1. 가격 데이터 로드
2. schema 검증
3. index 명칭 내부 변환
4. ETF/index close-to-close log return 계산
5. rolling corr/beta 계산
6. EWMA corr 보조 계산
7. active corr 선택
8. date별 corr 오름차순 market_rank 생성
9. market_score 생성
10. look-ahead 검증
11. market_rank.csv 저장
```

---

# 7. `02_make_flow_price_rank.py`

## 7.1 목적

주체별 수급이 ETF의 다음 시점 open-to-close 수익률에 대해 양/음의 예측력을 갖는지 rolling으로 분석하고, 그 예측 부호와 당일 수급 증감을 결합해 `flow_rank`를 만든다.

핵심:

\[
FlowScore_{i,t}=\sum_g \tilde{\gamma}_{i,g,t}\cdot SignalFlow_{i,g,t}
\]

여기서 \(g\)는 수급 주체다.

## 7.2 사용 데이터

입력 데이터:

```text
sector_all_merged.csv
sector_fund_flow.csv
```

가격 데이터 필수 컬럼:

```text
date
ticker
open
close
volume
amount
```

수급 데이터 필수 컬럼은 `FLOW_ACTOR_COLUMNS`에 포함된 actor 컬럼 전체다.

## 7.3 수급 정규화

`amount`는 백만원 단위 거래대금, `*_net_buy`는 천원 단위 순매수 금액으로 가정한다.

\[
flow\_ratio_{i,g,t}=\frac{netbuy_{i,g,t}}{amount_{i,t}\times1000+\epsilon}
\]

## 7.4 Flow z-score와 증감

오늘 flow가 abnormal한지를 보려면 baseline은 오늘 값을 포함하지 않아야 한다. 따라서 rolling mean/std는 `t-1`까지의 과거 window로 계산한다.

\[
\mu^{past}_{i,g,t}=Mean(flow\_ratio_{i,g,t-w:t-1})
\]

\[
\sigma^{past}_{i,g,t}=Std(flow\_ratio_{i,g,t-w:t-1})
\]

\[
z_{i,g,t}=\frac{flow\_ratio_{i,g,t}-\mu^{past}_{i,g,t}}{\sigma^{past}_{i,g,t}+\epsilon}
\]

구현 규칙:

```text
rolling_mean = rolling(flow_ratio).mean().shift(1)
rolling_std  = rolling(flow_ratio).std().shift(1)
flow_z[t] = (flow_ratio[t] - rolling_mean[t]) / rolling_std[t]
```

수급 증감:

\[
\Delta z_{i,g,t}=z_{i,g,t}-z_{i,g,t-1}
\]

당일 수급 signal:

\[
SignalFlow_{i,g,t}=0.7z_{i,g,t}+0.3\Delta z_{i,g,t}
\]

수급값 0은 실제 관측값으로 유지한다. 특정 actor의 rolling window 내 flow variance가 0이면 contribution을 0으로 처리한다. 이는 “해당 기간 예측 정보 없음”으로 해석한다.

## 7.5 주체별 예측력 추정

날짜 \(t\)의 주체별 predictive correlation은 다음 과거 pair만 사용한다.

\[
(z_{i,g,u}, r^{OC}_{i,u+1}), \quad u\le t-1
\]

여기서:

\[
r^{OC}_{i,u+1}=\log\frac{Close_{i,u+1}}{Open_{i,u+1}}
\]

중요:

```text
flow_rank[t]를 만들 때 z[t]와 r[t+1]의 관계는 아직 알 수 없다.
predictive corr[t]는 u <= t-1 샘플만 사용한다.
```

구현상 내부에서 `next_oc_ret[u]`를 계산할 수 있지만, `flow_rank[t]` 계산에는 rolling result를 한 날짜 shift하여 `u=t` label이 들어가지 않도록 한다.

Shrinkage:

\[
\tilde{\gamma}_{i,g,t}=\hat{\rho}_{i,g,t}\cdot\frac{n}{n+k}
\]

기본:

```text
k = 20
```

## 7.6 Flow score 해석

| 예측 부호 | 오늘 수급 | score 효과 |
|---:|---:|---:|
| \(\tilde{\gamma}>0\) | 매수 증가 | 상승 |
| \(\tilde{\gamma}>0\) | 매도 증가 | 하락 |
| \(\tilde{\gamma}<0\) | 매수 증가 | 하락 |
| \(\tilde{\gamma}<0\) | 매도 증가 | 상승 |

## 7.7 출력값: `flow_rank.csv`

기본 출력 컬럼:

```text
date
ticker
name
gics_sector
flow_score_raw
flow_score
flow_rank
flow_valid
flow_lookback
flow_obs
```

주체별 보조 컬럼:

```text
{actor}_flow_ratio
{actor}_flow_z
{actor}_flow_delta_z
{actor}_predictive_corr
{actor}_predictive_corr_shrunk
{actor}_predictive_obs
```

## 7.8 생성하고 사용할 함수

### `load_price_and_flow(price_path, flow_path)`

```text
가격 CSV 로드
수급 CSV 로드
date,ticker 기준 inner merge
정렬
```

### `validate_flow_schema(price_df, flow_df)`

```text
필수 컬럼 존재 여부
date,ticker 중복 여부
amount > 0 여부
actor column numeric 여부
```

### `compute_internal_next_open_to_close_label(df)`

내부용 label:

\[
next\_oc\_ret_{i,u}=\log\frac{Close_{i,u+1}}{Open_{i,u+1}}
\]

출력에는 저장하지 않는다.

### `normalize_actor_flows(df, actor_cols)`

생성:

```text
{actor}_flow_ratio
```

### `sanity_check_flow_units(df, actor_cols)`

```text
abs(flow_ratio)의 99% 분위수 점검
비정상적으로 큰 값 발생 시 warning 기록
```

### `compute_rolling_flow_zscore(df, actor_cols, lookback)`

과거 window를 shift하여 `t-1`까지의 mean/std로 z-score 생성.

```text
{actor}_flow_z
```

### `compute_flow_delta_z(df, actor_cols)`

```text
{actor}_flow_delta_z
```

### `compute_predictive_actor_corr(df, actor_cols, lookback, min_obs)`

```text
corr_at_t = corr(z[u], next_oc_ret[u]) using u <= t-1
```

### `shrink_predictive_corr(df, actor_cols, shrink_k)`

\[
\tilde{\gamma}=\hat{\rho}\cdot\frac{n}{n+k}
\]

### `compute_flow_score(df, actor_cols)`

\[
FlowScore_{i,t}=\sum_g \tilde{\gamma}_{i,g,t}\cdot(0.7z_{i,g,t}+0.3\Delta z_{i,g,t})
\]

생성:

```text
flow_score_raw
```

### `make_flow_rank(df)`

```text
각 date별 flow_valid ETF 선택
flow_score_raw 내림차순 rank
flow_rank 생성
flow_score 생성
```

### `validate_no_lookahead_flow(df)`

```text
flow_rank[t]에 next_oc_ret[t]가 들어가지 않았는지 확인
predictive corr 계산이 한 날짜 shift되었는지 확인
output에 next return label이 저장되지 않았는지 확인
```

## 7.9 처리 순서

```text
1. 가격 및 수급 CSV 로드
2. schema 검증
3. date,ticker 기준 병합
4. 내부용 next open-to-close label 계산
5. actor별 flow_ratio 계산
6. flow 단위 sanity check
7. actor별 rolling z-score 계산. baseline은 t-1까지 사용
8. actor별 delta z-score 계산
9. actor별 predictive corr 계산. 반드시 u <= t-1만 사용
10. predictive corr shrinkage 적용
11. 오늘 수급 signal과 predictive corr 부호/크기 결합
12. flow_score_raw 계산
13. date별 flow_rank 생성
14. flow_score 생성
15. look-ahead 검증
16. flow_rank.csv 저장
```

---

# 8. `03_make_rotation_rank.py`

## 8.1 목적

일별 ETF 수익률을 기반으로 순환매 rank를 만든다. 현재 버전에서는 수급 기반 rotation-flow는 사용하지 않는다.

## 8.2 사용 데이터

입력 데이터:

```text
sector_all_merged.csv
```

필수 컬럼:

```text
date
ticker
name
gics_sector
close
open
volume
amount
```

## 8.3 계산 로직

ETF close-to-close log return:

\[
ret\_1d_{i,t}=\log\frac{Close_{i,t}}{Close_{i,t-1}}
\]

같은 날짜 ETF universe 내부에서 cross-sectional z-score를 만든다.

\[
Z_{cs}(ret\_1d)_{i,t}=\frac{ret\_1d_{i,t}-\mu_t}{\sigma_t+\epsilon}
\]

기본 rotation score:

\[
rotation\_score\_raw_{i,t}=Z_{cs}(ret\_1d)_{i,t}
\]

Rank 방향:

```text
rotation_score_raw 높을수록 rotation_rank 상위
```

## 8.4 출력값: `rotation_rank.csv`

```text
date
ticker
name
gics_sector
ret_1d
rotation_score_raw
rotation_score
rotation_rank
rotation_valid
rotation_obs
```

## 8.5 생성하고 사용할 함수

### `load_price_data(path)`

CSV 로드 및 정렬.

### `validate_rotation_schema(df)`

```text
필수 컬럼 존재 여부
date,ticker 중복 여부
close 양수 여부
```

### `compute_close_to_close_returns(df)`

생성:

```text
ret_1d
```

### `cross_sectional_zscore(df, value_col, out_col)`

같은 날짜 ETF universe 내부 z-score 계산.

### `compute_rotation_score(df)`

생성:

```text
rotation_score_raw
```

### `make_rotation_rank(df)`

```text
각 date별 rotation_valid ETF 선택
rotation_score_raw 내림차순 rank
rotation_rank 생성
rotation_score 생성
```

### `validate_no_lookahead_rotation(df)`

```text
rotation_rank[t]가 close[t]까지만 사용했는지 확인
future return이 output에 포함되지 않았는지 확인
```

## 8.6 처리 순서

```text
1. 가격 CSV 로드
2. schema 검증
3. ETF별 close-to-close ret_1d 계산
4. date별 cross-sectional z-score 계산
5. rotation_score_raw 생성
6. rotation_score_raw 내림차순 rotation_rank 생성
7. rotation_score 생성
8. look-ahead 검증
9. rotation_rank.csv 저장
```

---

# 9. `04_merge_rank_panel.py`

## 9.1 목적

세 rank 파일을 하나의 canonical panel로 병합한다. A0~A6는 이 파일 하나만 읽는다.

## 9.2 사용 데이터

입력 파일:

```text
sector_all_merged.csv
market_rank.csv
flow_rank.csv
rotation_rank.csv
```

## 9.3 출력값: `rank_panel.csv`

```text
date
ticker
ETF_id
name
gics_sector
open
high
low
close
volume
amount
index_close

market_corr_active
market_corr_rolling20
market_corr_ewma20
market_beta_rolling20
market_score
market_rank
market_valid

flow_score_raw
flow_score
flow_rank
flow_valid

rotation_score_raw
rotation_score
rotation_rank
rotation_valid

tradable
all_rank_valid
```

금지:

```text
rank_panel.csv에는 next_open, next_close, next_return, winner_label 저장 금지
```

## 9.4 생성하고 사용할 함수

### `load_panel_inputs(price_path, market_path, flow_path, rotation_path)`

각 파일 로드.

### `standardize_price_panel(price_df)`

처리:

```text
kospi_close를 index_close로 rename
날짜 및 ticker 정렬
date,ticker 중복 확인
ticker는 내부 key로 유지
```

### `create_etf_id_mapping(price_df)`

전체 기간 ticker별 고정 익명 ID 생성.

```text
ETF_id = ETF_001, ETF_002, ...
asset_id = asset_001, asset_002, ...
```

`ETF_id`는 `rank_panel.csv`와 weights 복원용으로 저장한다. `asset_id`는 LLM prompt용이다.

### `merge_rank_files(price_df, market_df, flow_df, rotation_df)`

`date,ticker` 기준 병합.

### `compute_rank_scores_if_missing(df)`

rank score가 누락된 경우 재계산.

\[
score=1-\frac{rank-1}{N-1}
\]

### `make_tradable_flags(df)`

생성:

```text
tradable = open>0 and close>0 and volume>0 and amount>0
all_rank_valid = market_valid and flow_valid and rotation_valid
```

### `validate_rank_panel(df)`

검증:

```text
date,ticker 중복 없음
open/close/index_close 양수
rank 3종 존재
rank_panel에 future return/label 컬럼 없음
ticker가 ETF_id로 덮어써지지 않았음
ETF_id가 ticker별로 고정됨
```

### `save_rank_panel(df, out_path)`

출력 저장.

## 9.5 처리 순서

```text
1. price, market_rank, flow_rank, rotation_rank 로드
2. price panel에서 kospi_close를 index_close로 rename
3. ticker를 내부 key로 유지하고 ETF_id/asset_id mapping 생성
4. date,ticker 기준 병합
5. rank score 누락 여부 확인 및 보정
6. tradable, all_rank_valid flag 생성
7. future information 컬럼 없음 검증
8. rank_panel.csv 저장
9. id_mapping.csv 별도 저장
```

---

# 10. 공통 백테스트 엔진

A0~A6는 같은 백테스트 로직을 사용한다. 각 파일은 독립 실행형이지만, 내부 함수 구조는 동일해야 한다.

## 10.1 백테스트 입력

모든 실험군의 기본 입력:

```text
rank_panel.csv
```

필수 컬럼:

```text
date
ticker
ETF_id
open
close
market_rank
market_score
flow_rank
flow_score
rotation_rank
rotation_score
tradable
all_rank_valid
```

초기 자금:

```text
initial_nav = 1_000_000_000 KRW
```

백테스트 시작일은 `all_rank_valid=True` ETF가 존재하기 시작하는 최초 날짜다.

## 10.2 ETF sleeve weight 규칙

각 전략은 ETF sleeve 내부 weight만 결정한다.

\[
\sum_i w^{sleeve}_{i,t}=1
\]

v1에서는 `stock_ratio = 1.0`이다.

\[
PortfolioWeight_{i,t}=w^{sleeve}_{i,t}
\]

ETF weight cap은 적용하지 않는다.

## 10.3 `top_k` 규칙

A1~A6는 `top_k = 3, 5, 7, 10`으로 실행한다.

```text
top_k < 10:
    score 또는 LLM weight 기준 nonzero ETF 수를 최대 top_k개로 제한한다.

top_k = 10:
    hard filter를 적용하지 않는다.
    단, score-proportional allocation에서 score가 0인 ETF는 weight 0이 될 수 있다.
```

LLM 실험군 validator는 다음을 강제한다.

```text
nonzero_count <= min(top_k, N_valid)
sum(sleeve_weight) = 1
invalid asset_id weight = 0
```

A4는 다음 방식으로 top_k를 통제한다.

```text
1. A1 weight 생성
2. LLM weight 생성
3. 50:50 blend
4. final weight 상위 top_k만 유지. top_k=10이면 hard filter 없음
5. 다시 normalize_sleeve_weights 적용
```

A6는 expert ensemble 특성상 final nonzero ETF 수가 top_k를 초과할 수 있다. 따라서:

```text
A6의 top_k는 각 expert portfolio 생성에 적용한다.
A6 최종 blended portfolio는 effective_num_positions를 별도 기록한다.
```

`top_k`별 실행 결과는 suffix 방식으로 저장한다.

```text
A0:       out/A0/
A1~A5:    out/A1_k3/, out/A1_k5/, out/A1_k7/, out/A1_k10/ 등
A6-core:  out/A6_core_k3/, out/A6_core_k5/, out/A6_core_k7/, out/A6_core_k10/
A6-full:  out/A6_full_k3/, out/A6_full_k5/, out/A6_full_k7/, out/A6_full_k10/
```

A6-full은 같은 `top_k` suffix를 가진 expert 결과만 읽는다. 예를 들어 `A6_full_k5`는 `A1_k5`, `A2a_k5`, `A2b_k5`, `A3_k5`, `A4_k5`, `A5_k5`를 사용하고, A0만 공통 `out/A0/`를 사용한다.

## 10.4 거래비용

수수료는 매수와 매도 양쪽 모두에 one-way commission으로 계산한다.

\[
Commission=0.00015\times \sum_i |TargetValue_i-OldValue_i|
\]

수수료율:

```text
commission_rate = 0.00015
```

v1에서는 slippage, 세금, ETF 괴리, 호가 스프레드, 분배금 보정은 반영하지 않는다.

## 10.5 이벤트 루프

### Step 1. `t`일 종가

```text
rank_panel[t]를 사용해 target_sleeve_weight[t] 생성
```

### Step 2. `t+1`일 시가 전 기존 포지션 평가

기존 보유 수량 전체에 overnight PnL을 반영한다.

\[
PNL^{ON}_{t+1}=\sum_i q_{i,t}^{close}(Open_{i,t+1}-Close_{i,t})
\]

\[
NAV^{open,pre}_{t+1}=Cash_t+\sum_i q_{i,t}^{close}Open_{i,t+1}
\]

### Step 3. `t+1`일 시가 리밸런싱

수수료 반영 후에도 `stock_ratio = 1.0` 기준 target value와 commission을 fixed-point 방식으로 계산한다. 향후 외부 stock ratio를 사용할 경우 같은 식에서 `StockRatio`만 교체한다.

초기값:

\[
NAV^{open,post,(0)}=NAV^{open,pre}
\]

반복:

\[
TargetValue_i^{(k)}=StockRatio\times NAV^{open,post,(k)}\times w^{sleeve}_{i,t}
\]

\[
Commission^{(k)}=0.00015\times\sum_i |TargetValue_i^{(k)}-OldValue_i^{open}|
\]

\[
NAV^{open,post,(k+1)}=NAV^{open,pre}-Commission^{(k)}
\]

수렴 조건:

```text
commission_fixed_point_tol = 1e-10
commission_fixed_point_max_iter = 50
```

수렴 실패 시 마지막 iteration 값을 사용하고 warning log를 남긴다.

수렴 후:

\[
q_{i,t+1}^{new}=\frac{TargetValue_i}{Open_{i,t+1}}
\]

\[
Cash_{t+1}^{open,post}=(1-StockRatio)\times NAV^{open,post}
\]

### Step 4. `t+1`일 종가 평가

\[
PNL^{OC}_{t+1}=\sum_i q_{i,t+1}^{new}(Close_{i,t+1}-Open_{i,t+1})
\]

\[
NAV^{close}_{t+1}=Cash_{t+1}^{open,post}+\sum_i q_{i,t+1}^{new}Close_{i,t+1}
\]

## 10.6 공통 함수

### `load_rank_panel(path)`

CSV 로드, 날짜 변환, ticker string 변환, 정렬.

### `validate_backtest_panel(df)`

검증:

```text
필수 컬럼 존재
date,ticker 중복 없음
open/close 양수
rank/score 컬럼 존재
future return/label 컬럼 없음
ticker와 ETF_id 모두 존재
```

### `get_valid_universe(day_df)`

조건:

```text
tradable == True
all_rank_valid == True
market_score not null
flow_score not null
rotation_score not null
```

A0도 동일한 valid universe를 사용한다.

### `normalize_sleeve_weights(raw_weights, valid_tickers)`

```text
valid ETF 외 weight 제거
음수 weight 0 처리
합계가 양수면 합계 1로 정규화
합계가 0이면 valid ETF 동일비중 fallback
최종 sum=1 검증
```

### `apply_top_k_filter(weights, top_k)`

```text
top_k < 10이면 weight 상위 top_k만 유지 후 normalize
top_k = 10이면 hard filter 없음
```

### `rebalance_and_mark_to_market_next_day(state, target_sleeve_weights, price_t, price_t1, config)`

```text
기존 보유분 overnight PnL 계산
t+1 open에서 리밸런싱
commission 계산 및 차감
t+1 close까지 open-to-close PnL 계산
새 보유 수량 및 NAV 반환
```

### `compute_performance_metrics(daily_results)`

계산 지표:

```text
final_nav
total_return
CAGR
annual_vol
Sharpe
Sortino
MaxDD
Calmar
hit_ratio
avg_turnover_value
avg_turnover_ratio
total_turnover_value
total_turnover_ratio
total_commission
commission_drag
overnight_pnl_sum
open_to_close_pnl_sum
```

## 10.7 공통 출력

각 실험군은 다음 파일을 출력한다.

```text
daily_results.csv
weights.csv
summary.json
decision_log.jsonl
```

### `daily_results.csv`

```text
decision_date
execution_date
strategy
nav_close
nav_open_pre
nav_open_post
overnight_pnl
open_to_close_pnl
commission
turnover_value
turnover_ratio
daily_return
stock_ratio
fallback_used
effective_num_positions
```

### `weights.csv`

```text
decision_date
execution_date
ticker
ETF_id
sleeve_weight
portfolio_weight
```

### `summary.json`

```text
strategy
final_nav
total_return
cagr
annual_vol
sharpe
sortino
max_drawdown
calmar
hit_ratio
avg_turnover_value
avg_turnover_ratio
total_commission
fallback_rate
```

---

# 11. A0: `A0_equal_weight.py`

## 11.1 목적

동일비중 ETF sleeve baseline.

## 11.2 사용 데이터

```text
rank_panel.csv
```

사용 컬럼:

```text
date
ticker
open
close
tradable
all_rank_valid
```

## 11.3 의사결정 로직

각 날짜의 valid ETF universe에 동일비중을 부여한다.

\[
w_{i,t}^{A0}=\frac{1}{N_t}
\]

A0는 `top_k`와 무관하게 1회만 실행한다.

## 11.4 함수

### `generate_equal_weight(day_df)`

```text
valid universe 선택
모든 valid ETF에 1/N 부여
normalize_sleeve_weights로 검증
```

### `run_A0_backtest(panel)`

공통 백테스트 엔진 호출.

---

# 12. A1: `A1_rule_based_rank_allocator.py`

## 12.1 목적

세 rank score를 고정 가중치로 합성하는 deterministic baseline.

## 12.2 사용 데이터

```text
rank_panel.csv
```

사용 컬럼:

```text
market_score
flow_score
rotation_score
tradable
all_rank_valid
open
close
```

## 12.3 의사결정 로직

Composite score:

\[
Score_{i,t}=0.30\cdot market\_score_{i,t}+0.35\cdot flow\_score_{i,t}+0.35\cdot rotation\_score_{i,t}
\]

`top_k < 10`이면 상위 `top_k` ETF를 선택하고 score 비례 weight를 부여한다. `top_k = 10`이면 hard filter 없이 valid ETF 전체 후보에 대해 score 비례 weight를 부여한다. 이때 score가 0인 ETF는 weight 0이 될 수 있다.

\[
w_{i,t}=\frac{Score_{i,t}}{\sum_{j\in CandidateSet}Score_{j,t}}
\]

모든 score가 0이면 candidate set 동일비중으로 fallback한다.

## 12.4 함수

### `compute_composite_score(day_df)`

A1 composite score 계산.

### `select_candidate_set(day_df, score_col, top_k)`

```text
top_k < 10이면 score 내림차순 top-k 선택
top_k = 10이면 valid ETF 전체 후보 사용
```

### `generate_rule_based_weight(day_df)`

```text
valid universe 선택
composite_score 계산
candidate set 선택
score 비례 weight 생성
normalize_sleeve_weights 적용
```

### `run_A1_backtest(panel)`

공통 백테스트 엔진 호출.

---

# 13. LLM 공통 스키마 및 검증

A2a, A2b, A3, A4는 동일한 LLM infrastructure를 사용한다.

## 13.1 LLM 입력 allowlist

LLM prompt에 허용되는 필드:

```text
decision_step
is_initial_decision
asset_id
rank
rank_score
current_sleeve_weight
predefined constraints
A3/A4의 경우 policy pack 또는 base_sleeve_weight
```

금지 필드:

```text
ticker
ETF_id
name
gics_sector
index_close
raw price
raw volume
raw flow
actual date
future return
next open
next close
winner label
KOSPI
```

첫 decision date에는:

```text
current_sleeve_weight = 0 for all asset_id
is_initial_decision = true
```

이후:

```text
current_sleeve_weight = t close 기준 drift-adjusted sleeve weight
```

## 13.2 LLM 출력 JSON schema

A2a/A2b 최소 출력:

```json
{
  "decision_step": 123,
  "target_weights": [
    {
      "asset_id": "asset_001",
      "sleeve_weight": 0.35
    }
  ]
}
```

A3/A4 출력:

```json
{
  "decision_step": 123,
  "target_weights": [
    {
      "asset_id": "asset_001",
      "sleeve_weight": 0.35,
      "reason_codes": ["rank_alignment_strong"]
    }
  ],
  "portfolio_reason_codes": ["constraints_satisfied"]
}
```

Validator 강제 조건:

```text
JSON only
required fields 존재
asset_id valid
sleeve_weight numeric
sleeve_weight >= 0
sum(sleeve_weight) = 1
nonzero_count <= min(top_k, N_valid)
reason_codes는 A3/A4 enum 내부
invalid asset_id가 나오면 repair 또는 fallback
```

Repair는 1회만 허용한다. 실패하면 A1 fallback을 사용한다.

---

# 14. A2a: `A2a_llm_opaque_rank_allocator.py`

## 14.1 목적

LLM에게 세 rank signal의 의미를 숨기고, 순수 숫자 rank와 score만 제공했을 때 allocation 성능을 평가한다.

## 14.2 사용 데이터

```text
rank_panel.csv
```

LLM 입력 필드:

```text
asset_id
signal_1_rank = market_rank
signal_1_score = market_score
signal_2_rank = flow_rank
signal_2_score = flow_score
signal_3_rank = rotation_rank
signal_3_score = rotation_score
current_sleeve_weight
```

## 14.3 Prompt 원칙

A2a prompt는 매우 제한적이다.

```text
You are given three independent ranking signals.
Rank 1 is best.
Higher score is better.
Use only the provided ranks and scores.
Allocate 100% of the predefined asset sleeve across the provided asset_id universe.
Return JSON only.
```

넣지 않는 설명:

```text
market
flow
rotation
correlation
index
수급
순환매
policy rule
conflict matrix
```

## 14.4 함수

### `create_id_mapping(panel)`

전체 기간 ticker별 고정 `ETF_id`와 `asset_id` 생성.

### `build_opaque_packet(day_df, current_weights, id_mapping)`

A2a용 packet 생성.

### `build_opaque_prompt(packet)`

A2a minimal prompt 생성.

### `call_llm(prompt)`

LLM API 호출. Temperature는 0으로 고정한다.

### `parse_llm_json(response)`

응답 JSON parsing.

### `validate_llm_sleeve_weights(parsed, valid_asset_ids, top_k)`

공통 validator 적용.

### `repair_invalid_llm_output_once(...)`

1회 repair. 실패 시 A1 fallback.

### `map_asset_id_to_ticker(weights, id_mapping)`

LLM output을 실제 ticker weight로 복원.

### `generate_A2a_weight(day_df, current_weights)`

A2a decision 함수.

---

# 15. A2b: `A2b_llm_semantic_rank_allocator.py`

## 15.1 목적

LLM에게 세 rank의 최소 의미만 설명했을 때 allocation 성능을 평가한다.

A2a와의 차이:

```text
A2a: signal_1/2/3 의미 숨김
A2b: market/flow/rotation의 최소 의미 제공
```

## 15.2 사용 데이터

```text
rank_panel.csv
```

LLM 입력:

```text
asset_id
market_rank
market_score
flow_rank
flow_score
rotation_rank
rotation_score
current_sleeve_weight
```

## 15.3 Prompt 원칙

A2b에는 다음 최소 설명을 넣는다.

```text
market_rank:
    Lower correlation to a generic index is preferred.
    Rank 1 is best.

flow_rank:
    Investor-flow predictive signal rank.
    The rank already reflects whether each investor group historically had positive or negative predictive effect.
    Rank 1 is best.

rotation_rank:
    Return-based rotation rank.
    Rank 1 is best.

Use only the provided ranks and scores.
Do not use outside knowledge, market names, asset names, news, or memory.
Allocate 100% of the predefined asset sleeve.
Return JSON only.
```

A2b에는 policy pack, rank conflict matrix, reason code enum을 넣지 않는다.

## 15.4 함수

### `build_semantic_packet(day_df, current_weights, id_mapping)`

A2b용 packet 생성.

### `build_semantic_prompt(packet)`

세 rank의 최소 의미를 포함한 prompt 생성.

### `generate_A2b_weight(day_df, current_weights)`

A2b decision 함수.

---

# 16. A3: `A3_llm_policy_pack_allocator.py`

## 16.1 목적

LLM에게 세 rank의 최소 의미와 policy pack을 제공했을 때, A2b 대비 성과, 제약 준수, decision consistency가 개선되는지 평가한다.

## 16.2 사용 데이터

```text
rank_panel.csv
```

LLM 입력:

```text
asset_id
market_rank
market_score
flow_rank
flow_score
rotation_rank
rotation_score
current_sleeve_weight
```

## 16.3 Policy pack 내용

허용:

```text
rank 1 is best
market rank는 generic index와 낮은 correlation 선호
flow rank는 주체별 predictive sign을 이미 반영
rotation rank는 return-based rotation
rank alignment 원칙
rank conflict 처리 원칙
turnover 억제 원칙
분산 원칙
asset sleeve weight sum=1
JSON output schema
reason_code enum
```

금지:

```text
특정 시장명
특정 ETF명
특정 섹터 전망
뉴스
거장 persona
사전 시장지식
```

## 16.4 Rank conflict rule 예시

| market | flow | rotation | 처리 원칙 |
|---|---|---|---|
| strong | strong | strong | 높은 conviction |
| strong | weak | weak | 단독 market signal로 과도 배분 금지 |
| weak | strong | strong | selective allocation 가능 |
| strong | strong | weak | 중간 conviction |
| weak | strong | weak | watchlist 또는 낮은 비중 |
| weak | weak | strong | unsupported momentum 가능성. 낮은 비중 |
| weak | weak | weak | 제외 또는 최소 비중 |

## 16.5 Reason code enum

```text
rank_alignment_strong
market_rank_strong
flow_rank_strong
rotation_rank_strong
rank_conflict
market_rank_weak
flow_rank_weak
rotation_rank_weak
turnover_control
concentration_control
constraints_satisfied
fallback_required
```

## 16.6 함수

### `load_policy_pack()`

기본 policy pack 로드.

### `build_policy_packet(day_df, current_weights, id_mapping)`

A3용 anonymized packet 생성.

### `build_policy_prompt(packet, policy_pack)`

Policy pack 포함 prompt 생성.

### `parse_and_validate_reason_codes(parsed)`

reason code가 enum 안에 있는지 검증.

### `generate_A3_weight(day_df, current_weights)`

A3 decision 함수.

---

# 17. A4: `A4_rule_based_llm_blend.py`

## 17.1 목적

A1 rule-based weight와 A3-style LLM policy weight를 50:50 convex blend한다.

A4는 LLM에게 완전한 재량을 주는 모델이 아니라, rule-based baseline 위에 LLM 정책 판단을 절반만 반영하는 모델이다.

## 17.2 사용 데이터

```text
rank_panel.csv
```

A1용:

```text
market_score
flow_score
rotation_score
```

LLM용:

```text
asset_id
market_rank
market_score
flow_rank
flow_score
rotation_rank
rotation_score
current_sleeve_weight
base_sleeve_weight
```

## 17.3 의사결정 로직

A1 base weight:

\[
w^{A1}_{i,t}
\]

A3-style LLM policy weight:

\[
w^{LLM}_{i,t}
\]

A4 final weight:

\[
w^{A4}_{i,t}=0.5w^{A1}_{i,t}+0.5w^{LLM}_{i,t}
\]

\(\lambda\)는 최적화하지 않는다.

```text
lambda = 0.5 fixed
```

LLM output이 invalid이면:

\[
w^{A4}_{i,t}=w^{A1}_{i,t}
\]

A4 top-k 통제:

```text
blend 후 top_k < 10이면 final weight 상위 top_k만 유지하고 normalize한다.
top_k = 10이면 hard filter를 적용하지 않는다.
```

## 17.4 함수

### `generate_A1_base_weight(day_df)`

A1과 동일한 base weight 생성.

### `build_blend_llm_packet(day_df, current_weights, base_weights, id_mapping)`

LLM packet에 base weight 포함.

### `build_blend_policy_prompt(packet)`

A3 policy prompt와 유사하되, base weight를 참고하라고 지시한다.

### `generate_policy_llm_weight(day_df, current_weights, base_weights)`

LLM full sleeve weight 생성.

### `blend_weights(base_weights, llm_weights, lambda_value=0.5)`

\[
w^{final}=0.5w^{base}+0.5w^{llm}
\]

### `generate_A4_weight(day_df, current_weights)`

A4 decision 함수.

---

# 18. A5: `A5_bayesian_winner_loser_allocator.py`

## 18.1 목적

Regime 없이 세 rank score의 state bucket을 사용해 다음 구간 benchmark-beat 확률을 Bayesian posterior로 추정한다. LLM을 사용하지 않는다.

A5는 기존 “상위 30% cross-sectional winner” 정의를 사용하지 않는다. Winner는 benchmark인 generic `index`를 beat했는지 여부로 정의한다. 이 방식은 top 30% prior/base-rate 문제와 날짜별 winner 개수/tie 처리 문제를 제거한다.

## 18.2 사용 데이터

```text
rank_panel.csv
```

사용 컬럼:

```text
date
ticker
open
close
index_close
market_score
flow_score
rotation_score
tradable
all_rank_valid
```

## 18.3 Winner label

A5 내부에서만 다음 구간 close-to-close 수익률을 계산한다.

ETF next close-to-close return:

\[
R^{CC}_{i,t+1}=\frac{Close_{i,t+1}}{Close_{i,t}}-1
\]

Index next close-to-close return:

\[
R^{CC}_{index,t+1}=\frac{IndexClose_{t+1}}{IndexClose_t}-1
\]

Winner label:

\[
y_{i,t}=1 \quad \text{if } R^{CC}_{i,t+1} > R^{CC}_{index,t+1}
\]

Tie 처리:

```text
R_ETF == R_index이면 y=0으로 처리한다.
```

주의:

```text
winner_label은 rank_panel.csv에 저장하지 않는다.
A5 내부 학습용으로만 생성한다.
```

해석상 주의:

```text
A5 label은 benchmark-beat 예측용 close-to-close label이다.
실제 전략 PnL은 공통 백테스트 엔진의 t+1 open 리밸런싱 구조로 평가한다.
close-to-close label과 execution PnL은 동일 개념이 아니다.
```

이 선택은 사용자 의도상 허용한다. 비중 변경분은 t 장마감 후 결정되고, open-to-close는 시가 리밸런싱으로 인한 PnL 구조 분리를 위해 백테스트 엔진에서 별도 반영한다.

## 18.4 학습 데이터 경계

`t`일 장마감 decision에서 사용 가능한 feature:

```text
rank[t]
score[t]
close[t]
t일까지 계산 가능한 과거 정보
```

사용 불가:

```text
t+1 open
t+1 close
winner_label[t]
```

학습 가능한 sample:

```text
u <= t-1
```

왜냐하면 `winner_label[t-1]`은 `t`일 종가에 이미 확인되지만, `winner_label[t]`는 `t+1`일 종가 이후에야 확인되기 때문이다.

## 18.5 State bucket

각 score를 bucket으로 나눈다.

```text
top:    score >= 0.67
middle: 0.33 <= score < 0.67
bottom: score < 0.33
```

State key:

```text
M_{market_bucket}__F_{flow_bucket}__R_{rotation_bucket}
```

예:

```text
M_top__F_middle__R_top
```

## 18.6 Posterior

각 state \(b\)에 대해:

\[
\theta_b\sim Beta(\alpha_0,\beta_0)
\]

기본 prior:

\[
\alpha_0=1,\quad \beta_0=1
\]

Benchmark-beat base rate는 고정된 30%가 아니므로 Beta(1,1)의 weak neutral prior를 사용한다. 이후 state posterior는 expanding historical data와 global posterior shrinkage로 안정화한다.

Posterior:

\[
\theta_b|D_t\sim Beta(\alpha_0+wins_b,\beta_0+losses_b)
\]

Posterior mean:

\[
E[\theta_b|D_t]=\frac{\alpha_0+wins_b}{\alpha_0+\beta_0+wins_b+losses_b}
\]

표본이 적은 state는 global posterior와 shrinkage blend한다.

\[
p_{blend}=\lambda_b p_b+(1-\lambda_b)p_{global}
\]

\[
\lambda_b=\frac{n_b}{n_b+k}
\]

기본:

```text
k = 20
```

## 18.7 의사결정 로직

```text
1. t 시점 history u <= t-1로 posterior 계산
2. 현재 ETF별 state_key 생성
3. state_key별 posterior benchmark-beat probability 계산
4. top_k < 10이면 posterior score 상위 top-k ETF 선택
5. top_k = 10이면 hard filter 없이 valid ETF 전체 후보 사용
6. posterior score 비례 sleeve weight 생성
7. posterior score가 모두 0이면 candidate set 동일비중 fallback
8. normalize_sleeve_weights로 sum=1 검증
```

## 18.8 함수

### `compute_internal_next_cc_returns(panel)`

내부 label용 ETF/index next close-to-close return 계산.

### `make_benchmark_beat_labels(panel)`

`R_ETF_next_cc > R_index_next_cc` 기준 label 생성.

### `bucketize_scores(panel)`

market, flow, rotation score bucket 생성.

### `make_state_key(panel)`

state key 생성.

### `get_bayesian_history(panel, decision_date)`

```text
history = rows with label_date <= decision_date - 1
```

### `fit_state_posteriors(history)`

state별 wins/losses/posterior 계산.

### `score_current_day_with_posteriors(day_df, posterior_table)`

현재 ETF별 posterior benchmark-beat probability 계산.

### `generate_A5_weight(day_df, history_df)`

A5 target sleeve weight 생성.

## 18.9 출력

```text
out/A5_k{top_k}/daily_results.csv
out/A5_k{top_k}/weights.csv
out/A5_k{top_k}/summary.json
out/A5_k{top_k}/decision_log.jsonl
out/A5_k{top_k}/posterior_states.csv
```

---

# 19. A6: `A6_bayesian_online_expert_allocator.py`

## 19.1 목적

각 expert를 하나의 독립 투자 전략으로 보고, 매일 expert 성과를 이용해 D-ONS 기반 OBS-style mixture weight를 업데이트한다.

A6는 ETF-level Bayesian model이 아니라 strategy-level online expert ensemble이다. 여기서 사용하는 “Bayesian online” 표현은 OBS-style expert mixture의 개념적 framing이며, A6 update 자체는 Beta-Bernoulli success posterior가 아니다.

## 19.2 A6-core와 A6-full

### A6-core

메인 실험군이다.

Expert set:

```text
E0: A0_EQUAL_WEIGHT
E1: MARKET_ONLY
E2: FLOW_ONLY
E3: ROTATION_ONLY
E4: A1_RULE_COMPOSITE
E5: A5_BAYESIAN_WINNER
```

장점:

```text
LLM stochasticity 없음
API 장애 없음
rank signal별 생존력 추적 가능
A1/A5 대비 online ensemble 추가 가치 평가 가능
```

### A6-full

Meta-validation용 실험군이다.

Expert set:

```text
E0:  MARKET_ONLY
E1:  FLOW_ONLY
E2:  ROTATION_ONLY
E3:  A0_EQUAL_WEIGHT
E4:  A1_RULE_COMPOSITE
E5:  A2a_LLM_OPAQUE
E6:  A2b_LLM_SEMANTIC
E7:  A3_LLM_POLICY
E8:  A4_RULE_LLM_BLEND
E9:  A5_BAYESIAN_WINNER
```

A6-full은 A0~A5와 동일 선상에서 “최종 우승 전략”으로 제시하지 않는다. 목적은 다음을 검증하는 것이다.

```text
online mixture가 어떤 expert family를 신뢰하는가?
LLM 계열이 실제 성과가 있을 때 D-ONS expert probability를 받는가?
Bayesian A5가 rule/LLM보다 안정적인 expert인가?
```

A6-full에서는 A2a~A4 LLM을 다시 호출하지 않는다. 먼저 실행된 각 전략의 `weights.csv`와 `daily_results.csv`를 expert input으로 읽는다.

## 19.3 사용 데이터

A6-core 입력:

```text
rank_panel.csv
```

A6-full 입력:

```text
rank_panel.csv
out/A0/weights.csv
out/A0/daily_results.csv
out/A1_k{top_k}/weights.csv
out/A1_k{top_k}/daily_results.csv
out/A2a_k{top_k}/weights.csv
out/A2a_k{top_k}/daily_results.csv
out/A2b_k{top_k}/weights.csv
out/A2b_k{top_k}/daily_results.csv
out/A3_k{top_k}/weights.csv
out/A3_k{top_k}/daily_results.csv
out/A4_k{top_k}/weights.csv
out/A4_k{top_k}/daily_results.csv
out/A5_k{top_k}/weights.csv
out/A5_k{top_k}/daily_results.csv
```

## 19.4 Expert return과 거래비용 처리

A6에서 비용을 이중 차감하면 안 된다.

### Expert update용 return

A6 D-ONS update에 들어가는 expert return은 다음 중 하나다.

```text
A6-core:
    expert별 독립 shadow portfolio state를 실제 백테스트와 동일한 방식으로 운용해 계산한 net-of-commission wealth relative

A6-full:
    A0~A5 각 전략의 daily_results.csv에 저장된 net daily return 또는 net wealth relative
```

즉 D-ONS update에서는 이미 net-of-commission으로 계산된 expert gross return을 사용한다.

\[
r^{(e),net}_{t+1}=1+R^{(e),net}_{t+1}
\]

D-ONS update 단계에서 expert return에 추가 거래비용을 다시 차감하지 않는다.

### 실제 A6 portfolio 비용

A6가 생성한 blended portfolio는 별도 하나의 실제 포트폴리오다. 따라서 실제 A6 포트폴리오의 리밸런싱 비용은 공통 백테스트 엔진에서 한 번 반영한다.

```text
expert 비용을 합산하지 않는다.
A6 blended target weight의 실제 turnover에 대해서만 commission을 계산한다.
```

이 구조가 거래비용 과대계산을 막는다.

## 19.5 D-ONS 기반 OBS 업데이트

각 expert \(e\)의 net gross return을 \(r^{(e),net}_{t+1}\)라고 둔다.

\[
r^{(e),net}_{t+1}=1+R^{(e),net}_{t+1}
\]

Expert mixture weight \(\pi_t\)는 simplex 위에 있어야 한다.

\[
\sum_e \pi_{e,t}=1,\quad \pi_{e,t}\ge 0
\]

OBS loss:

\[
\ell_t(\pi)=-\log(\pi_t^\top r^{net}_{t+1})
\]

D-ONS는 realized net gross return vector에 대한 online convex optimization 문제로 처리한다.

기본 파라미터:

```text
dons_eta = 1.0
dons_gamma = 0.99
P0 = identity matrix
```

구현식은 다음으로 고정한다.

\[
g_t = \nabla \ell_t(\pi_t) = -\frac{r^{net}_{t+1}}{\pi_t^\top r^{net}_{t+1}}
\]

\[
H_t = \nabla^2 \ell_t(\pi_t)
    = \frac{r^{net}_{t+1}(r^{net}_{t+1})^\top}{(\pi_t^\top r^{net}_{t+1})^2}
\]

\[
P_{t+1}=(1-\gamma)P_0+\gamma P_t+H_t
\]

\[
\tilde{\pi}_{t+1}=\pi_t-\eta^{-1}P_{t+1}^{-1}g_t
\]

\[
\pi_{t+1}=\Pi_{\Delta}^{P_{t+1}}(\tilde{\pi}_{t+1})
\]

여기서 \(\Pi_{\Delta}^{P}\)는 simplex \(\Delta\) 위의 \(P\)-metric projection이다.

업데이트 구조:

```text
1. 현재 expert mixture probability pi_t로 A6 ETF sleeve weight 생성
2. 실제 A6 blended portfolio를 t+1 open에 체결하고 비용 1회 반영
3. expert별 shadow net gross return vector r_net[t+1] 계산 또는 외부 결과에서 로드
4. g_t와 H_t 계산
5. forgetting factor gamma로 P_{t+1} 업데이트
6. Newton step 후 P-metric simplex projection으로 pi_{t+1} 생성
```

최종 ETF sleeve weight:

\[
w^{A6}_{i,t}=\sum_e\pi_{e,t}w^{(e)}_{i,t}
\]

## 19.6 Index benchmark 리포팅

Index 수익률은 A6 업데이트 benchmark로 쓰지 않는다. 리포팅용으로만 저장한다.

Index close-to-close gross return:

\[
r^{index}_{t+1}=\frac{IndexClose_{t+1}}{IndexClose_t}
\]

주의:

```text
ETF 전략은 t+1 open 리밸런싱과 t+1 close 평가 구조다.
index benchmark는 index open 데이터가 없으므로 close-to-close 기준으로만 계산한다.
따라서 index-relative metric은 보조 진단용이지 동일 체결 기준의 직접 성과 비교가 아니다.
```

저장 항목:

```text
index_gross_return_close_to_close
a6_log_wealth
a6_index_relative_log_wealth
expert_index_relative_gross_return_close_to_close
```

## 19.7 함수

### `make_equal_weight_expert(day_df)`

A0 expert weight 생성.

### `make_single_rank_expert(day_df, score_col)`

`market_score`, `flow_score`, `rotation_score` 중 하나만 사용해 top-k score proportional weight 생성.

### `make_composite_expert(day_df)`

A1 composite expert 생성.

### `make_A5_expert(day_df, history_df)`

A5 Bayesian benchmark-beat expert 생성.

### `load_external_expert_weights_and_returns(paths)`

A6-full에서 A0~A5 weights와 net daily return을 읽는다.

### `initialize_dons_state(expert_names)`

expert mixture probability와 D-ONS second-order state 초기화.

### `compute_expert_mixture_probs(dons_state)`

현재 D-ONS state의 expert mixture probability 반환.

### `blend_expert_weights(expert_weights, expert_probs)`

expert portfolio들을 D-ONS expert probability로 혼합.

### `initialize_shadow_expert_states(expert_names)`

A6-core에서 expert별 독립 shadow portfolio state 생성.

필요 상태:

```text
expert별 NAV
expert별 보유 수량 q_i
expert별 전일 close 기준 평가값
expert별 current sleeve weight
```

### `rebalance_and_mark_to_market_expert_shadow_state(expert_state, expert_target_weights, price_t, price_t1, config)`

A6-core에서 각 expert가 독립 운용되었다고 가정한 net gross return 계산.

반영 항목:

```text
기존 expert 보유분 overnight PnL
expert별 t+1 open 리밸런싱
expert별 commission 1회
expert별 t+1 open-to-close PnL
```

### `update_dons_state(dons_state, expert_net_gross_returns)`

D-ONS gradient/hessian update와 simplex projection으로 다음 expert mixture probability 계산.

### `generate_A6_weight(day_df, history_df, expert_state)`

A6 target sleeve weight 생성.

## 19.8 처리 순서

```text
1. rank_panel.csv 로드
2. D-ONS expert state 초기화
3. A6-core는 expert별 shadow portfolio state 초기화
4. 각 decision date t 반복
5. day_df[t]에서 valid universe 선택
6. expert별 sleeve portfolio 생성 또는 외부 weights 로드
7. 현재 D-ONS state로 expert mixture probability 계산
8. expert portfolio를 D-ONS probability-weighted blend
9. A6 target sleeve weight 생성
10. 실제 A6 blended portfolio를 t+1 open에 체결하고 비용 1회 반영
11. expert별 net gross return 계산 또는 로드
12. D-ONS state update. 추가 비용 차감 없음
13. weights, daily_results, expert_probs, expert_net_gross_returns, log wealth 저장
```

## 19.9 출력

```text
out/A6_core_k{top_k}/daily_results.csv
out/A6_core_k{top_k}/weights.csv
out/A6_core_k{top_k}/summary.json
out/A6_core_k{top_k}/decision_log.jsonl
out/A6_core_k{top_k}/expert_probs.csv
out/A6_core_k{top_k}/expert_net_gross_returns.csv
out/A6_core_k{top_k}/a6_log_wealth.csv
out/A6_core_k{top_k}/a6_index_relative_log_wealth.csv

out/A6_full_k{top_k}/daily_results.csv
out/A6_full_k{top_k}/weights.csv
out/A6_full_k{top_k}/summary.json
out/A6_full_k{top_k}/decision_log.jsonl
out/A6_full_k{top_k}/expert_probs.csv
out/A6_full_k{top_k}/expert_net_gross_returns.csv
out/A6_full_k{top_k}/a6_log_wealth.csv
out/A6_full_k{top_k}/a6_index_relative_log_wealth.csv
```

A6 지표 명칭은 `posterior` 대신 `mixture weight` 또는 `expert probability`를 사용한다.

```text
best_expert_by_average_mixture_weight
final_expert_mixture_weight
average_expert_probability
expert_probability_entropy
```

---

# 20. Look-ahead 방지 체크리스트

## 20.1 Rank 생성 단계

| 파일 | 방지 규칙 |
|---|---|
| `01_make_market_corr_rank.py` | `market_corr[t]`는 `t`일까지의 close return만 사용 |
| `02_make_flow_price_rank.py` | `flow_z[t]`의 baseline은 `t-1`까지 사용. `predictive_corr[t]`는 `u <= t-1` 샘플만 사용 |
| `03_make_rotation_rank.py` | `rotation_rank[t]`는 `close[t]`까지만 사용 |
| `04_merge_rank_panel.py` | future return, next open/close, winner label 저장 금지 |

## 20.2 LLM 단계

LLM prompt에는 다음을 넣지 않는다.

```text
actual date
future return
next open
next close
winner label
ETF name
ticker
ETF_id
gics_sector
KOSPI
index price
raw price series
news
```

LLM prompt에는 다음만 허용한다.

```text
decision_step
asset_id
rank
rank_score
current_sleeve_weight
predefined constraints
A3/A4의 경우 policy pack 또는 base_sleeve_weight
```

## 20.3 Backtest 단계

```text
t일 장마감 target weight 생성
t+1일 시가 체결
t+1일 종가 평가
```

금지:

```text
t일 rank로 t일 close 체결 가정
t일 decision에서 t+1 price 사용
A5 posterior에서 label[t] 사용
```

A5에서는:

```text
t decision 시점 학습 가능 sample = u <= t-1
label[t]는 t+1 close 이후에만 관측 가능
```

---

# 21. Survivorship bias 방지 및 한계

## 21.1 구현상 방지

```text
1. 날짜별 valid universe 사용
2. 상장 전 ETF universe 제외
3. rank valid=False ETF 제외
4. 누락 가격 forward-fill 금지
5. A0~A6 모두 동일 valid universe 사용
```

## 21.2 남는 한계

첨부 데이터가 프로젝트 universe에 포함된 10개 ETF만 제공한다면, 폐지되었거나 제외된 ETF에 대한 survivorship bias는 완전히 제거할 수 없다.

완전한 제거를 원하면 다음 데이터가 필요하다.

```text
전체 ETF 상장/폐지 이력
상장폐지 ETF 가격 및 수급 데이터
날짜별 실제 investable universe
```

결과에는 다음 단서를 붙인다.

```text
This experiment controls survivorship bias within the observed ETF universe, but cannot eliminate survivorship bias from ETFs absent from the source dataset.
```

---

# 22. 최종 성과 평가 지표

## 22.1 수익률 지표

```text
final_nav
total_return
CAGR
annualized_volatility
Sharpe
Sortino
MaxDD
Calmar
hit_ratio
```

## 22.2 비용 및 회전율

```text
avg_turnover_value
avg_turnover_ratio
total_turnover_value
total_turnover_ratio
total_commission
commission_drag
average_holding_period_proxy
```

## 22.3 PnL 분해

```text
overnight_pnl_sum
open_to_close_pnl_sum
overnight_pnl_share
open_to_close_pnl_share
```

## 22.4 LLM 운영 지표

A2a, A2b, A3, A4에서 계산한다.

```text
json_validity_rate
schema_compliance_rate
constraint_violation_rate
repair_rate
fallback_rate
average_prompt_tokens
average_response_tokens
```

## 22.5 A6 전용 지표

```text
average_expert_probability
final_expert_mixture_weight
expert_probability_entropy
best_expert_by_average_mixture_weight
best_expert_by_realized_return
expert_turnover
a6_index_relative_log_wealth
```

---

# 23. 최종 실험 비교표

메인 실험군:

```text
A0  Equal Weight, top_k 무관 1회 실행
A1  Rule-Based Composite, top_k = 3/5/7/10
A2a LLM Opaque Rank, top_k = 3/5/7/10
A2b LLM Semantic Rank, top_k = 3/5/7/10
A3  LLM Policy Pack, top_k = 3/5/7/10
A4  Rule + LLM Blend, lambda=0.5, top_k = 3/5/7/10
A5  Bayesian Benchmark-Beat Winner/Loser, top_k = 3/5/7/10
A6-core D-ONS OBS Expert, top_k = 3/5/7/10
```

보조 실험군:

```text
A6-full Meta-Validation Expert Ensemble
```

A6-full은 메인 성과 비교에서는 별도로 표시한다. A6-full은 “어떤 전략군을 D-ONS expert probability가 신뢰했는가”를 보는 검증 실험이지, 단순한 최종 우승 후보가 아니다.

---

# 24. 개발 순서

권장 구현 순서:

```text
1. 01_make_market_corr_rank.py
2. 02_make_flow_price_rank.py
3. 03_make_rotation_rank.py
4. 04_merge_rank_panel.py
5. 공통 백테스트 엔진 구현
6. A0 구현
7. A1 구현
8. A5 구현
9. A2a 구현
10. A2b 구현
11. A3 구현
12. A4 구현
13. A6-core 구현
14. A6-full 구현
15. 전체 결과표 생성
16. leakage / survivorship / LLM anonymization audit
```

이 순서가 좋은 이유:

```text
A0/A1/A5는 LLM 없이 재현 가능한 baseline이다.
LLM 실험군 전에 rank_panel과 backtest engine을 먼저 고정해야 한다.
A6는 A0~A5 결과를 expert portfolio로 사용할 수 있으므로 마지막에 구현한다.
A6는 Beta-Bernoulli success posterior가 아니라 D-ONS 기반 OBS update로 구현한다.
```

---

# 25. 필수 감사 로그

각 실행은 다음 metadata를 저장한다.

```text
run_id
strategy
top_k
input_file_hash
rank_panel_hash
config_hash
prompt_hash
model_id
llm_provider
temperature
commission_rate
stock_ratio
initial_nav
created_at
```

LLM 실험군은 추가로 다음을 저장한다.

```text
asset_id mapping hash
raw prompt hash
raw response hash
parsed JSON
validation result
repair used
fallback used
```

실제 ETF명과 ticker가 포함된 mapping 파일은 LLM 로그와 분리하여 저장한다.

---

# 26. 최종 핵심 요약

본 프로젝트는 다음 원칙을 지킨다.

```text
1. 세 rank signal만 사용한다.
2. Regime, 뉴스, 외부 데이터는 사용하지 않는다.
3. Market rank는 rolling correlation 낮은 ETF를 선호한다.
4. Flow rank는 actor별 예측 부호와 수급 증감을 결합한다.
5. Rotation rank는 일별 수익률 기반이다.
6. 모든 rank는 t일 장마감 기준으로 계산된다.
7. 모든 전략은 t+1일 시가에 체결된다.
8. 기존 보유분 overnight PnL과 신규 포지션 open-to-close PnL을 모두 반영한다.
9. ETF sleeve weight는 항상 합계 1이다.
10. 초기 NAV는 10억원이다.
11. 수수료는 매수/매도 모두 0.015% one-way다.
12. A0는 top_k와 무관하게 1회만 실행하고, A1~A6는 top_k 3/5/7/10을 실행한다.
13. top_k=10은 hard filter를 적용하지 않는다. 다만 score가 0이면 weight 0 가능하다.
14. LLM 입력에서는 ETF명, ticker, 섹터명, KOSPI, 실제 날짜를 숨긴다.
15. 내부에서는 ticker를 유지하고, LLM prompt에는 asset_id만 노출한다.
16. A2a/A2b/A3/A4는 LLM 정보 제공 수준 차이를 비교한다.
17. A5는 ETF-level Bayesian benchmark-beat posterior다.
18. A6는 D-ONS 기반 strategy-level OBS expert ensemble이다.
19. A6 update에는 expert net-of-commission return을 사용하며 추가 비용을 차감하지 않는다.
20. 실제 A6 blended portfolio의 리밸런싱 비용은 별도로 한 번만 반영한다.
21. Look-ahead와 survivorship bias를 가능한 범위에서 명시적으로 통제한다.
```

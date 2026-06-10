---
title: "ALE/Breakout-v5에서의 심층 강화학습: 알고리즘 비교와 Ablation 분석"
running_head: "BREAKOUT 심층 강화학습"
format: APA 7th Edition
language: Korean
---

<!-- ================================================================
     APA 7판 형식 안내 (Word 작성 시 적용)
     - 전체 본문: 12pt Times New Roman, 더블 스페이싱, 2.54cm 여백
     - 제목 페이지: Running head 생략(7판), 페이지 번호 우상단
     - 단락 첫 줄 들여쓰기: 1.27cm (Tab 1회)
     - 표·그림: 본문 인용 후 해당 위치에 삽입 (APA 7판 인라인 허용)
     - 참고문헌: 매달리기 들여쓰기(Hanging indent) 1.27cm
================================================================ -->


# ALE/Breakout-v5에서의 심층 강화학습: 알고리즘 비교와 Ablation 분석

이찬휘

포항공과대학교 (POSTECH)

---
<!-- 제목 페이지 끝. 다음 페이지부터 Abstract -->

# 초록 (Abstract)

본 연구는 Atari 게임 환경 ALE/Breakout-v5에서 다섯 가지 심층 강화학습 알고리즘—DQN (Mnih et al., 2015), Double DQN (van Hasselt et al., 2016), Dueling DQN (Wang et al., 2016), PPO (Schulman et al., 2017), A2C (Mnih et al., 2016)—을 비교하고, 단일 변수(one-knob-at-a-time) 원칙에 따른 하이퍼파라미터 ablation을 수행하여 강화학습 이론과 실제 결과의 간극을 분석하였다. 알고리즘 비교는 최종 성능 외에도 샘플 효율(AUC, 임계 도달 스텝), 안정성(tail std, drawdown), 행동 형태(타임캡 도달률 cap_rate, 왜도), 연산 비용, 통계적 유의성의 다기준으로 수행하였다. 시드 7/77/777에 대한 교차 시드 분석에서 10M 스텝 기준 알고리즘 순위는 A2C(교차 시드 평균 207점) > Dueling DQN(127점) > PPO(103점) > DQN(55점) ≈ Double DQN(50점)으로 나타났다. Ablation 분석에서는 DQN 학습률을 1.0e-4에서 1.5e-4로 조정하면 성능이 2.5배 향상되는 반면, A2C에서 advantage 정규화를 활성화하면 성능이 97% 폭락하는 대조적인 결과가 확인되었다. 학습 예산을 50M 스텝으로 확장하면 off-policy DQN(369점)이 A2C(378점) 및 PPO(338점)에 근접하여, 단기 예산에서의 on-policy 우위가 예산 의존적임을 확인하였다. 전체 분석은 5개 알고리즘을 최대 4개 학습 예산(1M/2M/10M/50M 스텝) 및 최대 3개 시드에 걸쳐 수행된 **약 54회의 독립 학습 실험** 결과를 기반으로 하며, DQN(6종) · PPO(10종) · A2C(3종) 합계 **19개의 ablation 변형 실험**(설계 27종 중 완료 19종)을 포함한다. 학습에 사용된 총 환경 스텝은 약 5억 스텝(≈ 500M env steps)이며, 논문 본문의 표·그림 외 원시 데이터와 분석 코드는 리포지토리의 [scripts/](../scripts/)와 [src/analysis.py](../src/analysis.py)를 통해 완전히 재현 가능하다.

*키워드*: 심층 강화학습, Atari Breakout, DQN, PPO, A2C, 하이퍼파라미터 민감도, 샘플 효율

---

# 1. 서론

Atari 게임 Breakout은 강화학습(Reinforcement Learning, RL) 연구의 표준 벤치마크 환경 중 하나이다 (Mnih et al., 2015). 에이전트는 픽셀 이미지만을 입력으로 받아 공을 벽돌에 맞히는 전략을 스스로 학습해야 하므로, 시각적 인식, 장기 보상 할인, 탐색-이용(exploration-exploitation) 균형 등 강화학습의 핵심 문제들이 모두 등장한다.

본 프로젝트의 목표는 단순히 높은 점수를 달성하는 것이 아니라, 다양한 알고리즘과 하이퍼파라미터 실험을 통해 강화학습 이론과 실전 결과의 차이를 분석하는 것이다. 이를 위해 동일한 환경과 전처리 파이프라인 위에서 다섯 가지 알고리즘을 공정하게 비교하고, 각 알고리즘의 핵심 하이퍼파라미터를 체계적으로 변화시키는 ablation 실험을 수행하였다. 특히 단일 점수 비교의 한계를 보완하기 위해, 다기준 평가 체계와 다중 시드(7/77/777)를 사용한 교차 시드 집계를 도입하였다.

## 1.1 라이브러리 활용 방식

본 프로젝트는 Stable-Baselines3 (SB3)를 기반으로 하되, 단순 실행에 머물지 않고 원본 기여(original contributions)를 추가하였다. DQN은 SB3의 `DQN` 클래스를 그대로 사용하였으나, Double DQN은 SB3 `DQN`을 상속하여 `train()` 메서드를 오버라이드함으로써 Double Q-learning 타깃을 직접 구현하였다. Dueling DQN은 SB3의 `QNetwork`를 상속하여 Value/Advantage 분기 네트워크로 교체하고 `DuelingCnnPolicy`를 정의하였다. PPO와 A2C는 SB3의 검증된 구현을 그대로 활용하되 YAML 설정 파일에서 하이퍼파라미터를 주입하였다. 즉, DQN 계열에서는 핵심 수식(Double Q-learning target, Dueling head)을 직접 구현하여 이론적 이해를 코드로 표현하였다.

---

# 2. 환경 및 전처리

## 2.1 환경 설정

실험 환경은 `ALE/Breakout-v5`로, `gymnasium[atari]==1.3.0`, `ale-py==0.11.2`, `autorom[accept-rom-license]==0.6.1` 패키지를 사용하였다. ALE 설정은 다음과 같다: `frameskip=1` (SB3의 `MaxAndSkip` 래퍼가 frame skip을 직접 처리), `repeat_action_probability=0.0` (무작위 행동 반복 비활성화), `full_action_space=False` (Breakout의 필수 행동 4개만 사용).

## 2.2 전처리 파이프라인

SB3의 `make_atari_env()`는 내부적으로 `AtariWrapper`를 적용하며, 다음의 전처리 단계로 구성된다.

원본 RGB 프레임(210×160×3)에 `MaxAndSkip(4)`를 적용하여 4 프레임마다 최대 픽셀을 취하고, `NoopReset`으로 에피소드 시작 시 0~30회 무작위 NOP을 실행하여 초기 상태를 다양화한다. 학습 시에는 `EpisodicLife`로 생명 소모 시 에피소드를 종료 처리하며, `FireReset`으로 FIRE 버튼 조작을 처리한다. 이후 `WarpFrame(84×84)`로 그레이스케일 변환 및 리사이즈, `ClipReward`로 보상을 [-1, +1]로 클리핑, `VecFrameStack(4)`로 연속 4 프레임을 스택한다. 최종 입력 형태는 (4, 84, 84)이다.

## 2.3 학습 환경과 평가 환경의 차이

학습 환경과 평가 환경은 의도적으로 분리하였다 (표 1 참조). 학습 시 생명 소모를 에피소드 종료로 처리하는 것은 에이전트가 "죽지 않는 것"이 즉각적인 부정적 결과임을 더 빠르게 학습하게 하는 일반적인 Atari 학습 기법이다. 그러나 평가는 실제 게임 완주 점수로 측정하므로 두 환경을 분리하였다.

**표 1**

*학습 환경과 평가 환경의 주요 차이*

| 항목 | 학습 환경 | 평가 환경 |
|---|---|---|
| `terminal_on_life_loss` | True (생명 소모 = 에피소드 종료) | False (게임 전체가 1 에피소드) |
| 보상 클리핑 | [-1, +1] | 없음 (실제 점수) |
| 병렬 env 수 | 8 | 1 |

---

# 3. 알고리즘 구현

## 3.1 DQN (Deep Q-Network)

DQN은 Q-함수를 신경망으로 근사하고, Experience Replay와 Target Network로 학습을 안정화한다 (Mnih et al., 2015). 표준 DQN의 Q-learning 업데이트 타깃은 다음과 같다.

$$y_i = r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a')$$

핵심 안정화 기법은 두 가지이다. 첫째, Experience Replay는 과거 전이 $(s, a, r, s')$를 Replay Buffer에 저장하고 미니배치로 샘플링하여 시간적 상관관계를 제거한다. 둘째, Target Network는 Q-learning 타깃 계산에 주기적으로 복사된 별도 네트워크를 사용하여 타깃의 변동성을 줄인다.

## 3.2 Double DQN

표준 DQN은 $\max_{a'} Q(s', a')$를 타깃 네트워크 하나로 계산하므로 최대치를 과대 추정하는 경향이 있다 (van Hasselt et al., 2016). Double DQN은 행동 선택과 Q값 평가를 분리하여 이 편향(bias)을 줄인다.

$$y_i = r + \gamma \cdot Q_{\text{target}}\!\left(s',\; \arg\max_{a'} Q_{\text{online}}(s', a')\right)$$

본 프로젝트에서는 SB3의 `DQN`을 상속하여 `train()` 메서드를 오버라이드하였다. 기존의 `max_a' Q_target(s', a')` 계산을 온라인 네트워크의 행동 선택(`next_q_online.argmax()`)과 타깃 네트워크의 Q값 평가(`next_q_target.gather()`)로 분리하는 것이 Double DQN 구현의 핵심이다.

## 3.3 Dueling DQN

Dueling DQN은 Q-함수를 상태 가치 $V(s)$와 행동 어드밴티지 $A(s,a)$의 합으로 분해한다 (Wang et al., 2016).

$$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)$$

평균을 빼는 것은 $V(s)$와 $A(s,a)$의 정체성 모호성(identifiability)을 해소하기 위한 것이다. SB3에는 Dueling 네트워크가 내장되어 있지 않으므로, 본 프로젝트에서는 SB3의 `QNetwork`를 직접 상속하여 Dueling head를 구현하고 `DuelingCnnPolicy`를 새로 정의하였다. 아키텍처는 NatureCNN 공유 특징 추출기에서 `value_net`(FC(512) → Linear(1))과 `advantage_net`(FC(512) → Linear(n_actions))으로 분기한다.

## 3.4 PPO (Proximal Policy Optimization)

PPO는 On-policy 정책 그래디언트 알고리즘으로, 정책 업데이트가 너무 크게 변하지 않도록 Clip 제약을 걸어 학습을 안정화한다 (Schulman et al., 2017). 본 실험에서는 Clip 방식을 사용하였다.

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

여기서 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$는 정책 비율, $\hat{A}_t$는 GAE (Schulman et al., 2016)로 추정한 어드밴티지이다. SB3의 PPO 구현을 그대로 사용하되, YAML 설정 파일에서 모든 하이퍼파라미터를 주입하였다.

## 3.5 A2C (Advantage Actor-Critic)

A2C는 Actor(정책)와 Critic(가치 함수)을 동시에 학습하는 On-policy 알고리즘으로, A3C (Mnih et al., 2016)의 동기 버전이다. 여러 환경을 동기적으로 실행하여 경험을 수집한다.

$$\nabla_\theta J(\theta) \approx \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]$$

$$\hat{A}_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)$$

SB3 기본 A2C는 옵티마이저로 RMSProp을 사용한다. 이는 A3C 원 논문의 설정을 따른 것으로, 비정상적(non-stationary) 그래디언트 환경에서 적응적 학습률을 제공한다.

---

# 4. 실험 설정

## 4.1 공통 설정

모든 알고리즘에 공통적으로 적용한 설정은 표 2와 같다. 특징 추출기로는 Mnih et al. (2015)의 NatureCNN을 사용하였다.

**표 2**

*모든 알고리즘에 공통 적용된 실험 설정*

| 항목 | 값 |
|---|---|
| 총 학습 스텝 | 10,000,000 |
| 시드 | 7 (교차 시드: 7 / 77 / 777) |
| 특징 추출기 | NatureCNN (Mnih et al., 2015) |
| 학습용 병렬 env | 8 |
| 평가용 env | 1 |
| 평가 에피소드 수 | 100 |

## 4.2 알고리즘별 하이퍼파라미터

DQN/Double DQN/Dueling DQN의 하이퍼파라미터는 표 3에, PPO는 표 4에, A2C는 표 5에 정리하였다.

**표 3**

*DQN 계열 공통 하이퍼파라미터*

| 하이퍼파라미터 | 값 | 근거 |
|---|---|---|
| `learning_rate` | 1.0e-4 | SB3 Atari DQN 기본값 |
| `buffer_size` | 250,000 | VRAM/RAM 제약 으로 인한 절충값 (Mnih et al., 2015의 1M 대비 축소) |
| `batch_size` | 32 | Nature DQN 논문 설정값 |
| `train_freq` | 4 steps | 4 스텝마다 1회 업데이트 |
| `target_update_interval` | 1,000 | SB3 기본값 |
| `exploration_fraction` | 0.1 | 전체의 10%에서 ε 1.0→0.01로 감소 |
| `gamma` | 0.99 | 표준값 |
| `max_grad_norm` | 10.0 | — |

**표 4**

*PPO 하이퍼파라미터*

| 하이퍼파라미터 | 값 | 근거 |
|---|---|---|
| `learning_rate` | 2.5e-4 | Schulman et al. (2017) Atari PPO 표준 |
| `n_steps` | 128 | 128 × 8 env = 1,024 steps/update |
| `batch_size` | 256 | — |
| `n_epochs` | 4 | SB3 Atari 기본값 |
| `gamma` | 0.99 | 표준값 |
| `gae_lambda` | 0.95 | GAE 원 논문 (Schulman et al., 2016) 설정값 |
| `clip_range` | 0.1 | SB3 Atari 기본값|
| `ent_coef` | 0.01 | 탐색 장려를 위한 엔트로피 보너스 |
| `vf_coef` | 0.5 | 가치 손실 가중치 |
| `max_grad_norm` | 0.5 | — |

**표 5**

*A2C 하이퍼파라미터*

| 하이퍼파라미터 | 값 | 근거 |
|---|---|---|
| `learning_rate` | 7.0e-4 | A3C/A2C 표준 RMSProp LR |
| `n_steps` | 5 | 단기 n-step return (A3C 논문 방식) |
| `gamma` | 0.99 | 표준값 |
| `gae_lambda` | 1.0 | λ=1 = 순수 n-step return (GAE 미사용) |
| `ent_coef` | 0.01 | 엔트로피 보너스 |
| `vf_coef` | 0.25 | SB3 A2C Atari 기본값 |
| `use_rms_prop` | True | A3C 원 논문의 RMSProp 사용 |
| `normalize_advantage` | False | 기본값 |
| `rms_prop_eps` | 1.0e-5 | — |

---

# 5. 분석 방법론 및 종합 결과

## 5.1 분석 데이터와 파이프라인

재학습·재평가 없이, 학습과 평가 단계가 이미 남긴 산출물만을 읽어 분석하였다 (표 6 참조). 파이프라인은 `discover_runs()`가 `experiments/` 하위 모든 실행을 메타데이터와 함께 수집하고, `dedup_latest()`가 동일 (알고리즘, 변형, 시드, 예산) 설정의 중복 실행을 제거한 뒤, 8개 분석 축(알고리즘·시드·예산·하이퍼파라미터 응답·샘플 효율·연산·행동 형태·유의성)을 계산하는 구조이다.

**표 6**

*분석에 사용된 산출물과 지표*

| 산출물 | 내용 | 사용 지표 |
|---|---|---|
| `eval/summary.json` | 최종 정책의 100 에피소드 평가 통계 | 최종 성능 |
| `eval/episodes_*.csv` | 100 에피소드의 개별 보상·길이 | 분포 형태, 유의성 |
| `eval/evaluations.npz` | 학습 중 주기적 평가 곡선 | 샘플 효율, 안정성 |
| `monitor.csv` | 학습 에피소드 길이·시각 | 연산 비용(시간/FPS) |
| `config.yaml` | 알고리즘·하이퍼파라미터·budget·seed | 그룹화/식별 |

## 5.2 평가 기준

단일 최종 평균 점수만으로 순위를 매기지 않았다. 높은 평균이라도 (a) 시드 운, (b) 큰 후반 진동, (c) 점수가 아니라 시간만 끄는 생존(타임캡 도달)으로 얻은 것이라면 약한 증거로 보았다. 사용한 지표는 다음과 같다: 최종 성능(`final_mean`, `median`, `final_ci95`), 샘플 효율(`auc_mean_reward`, `steps_to_threshold`), 안정성(`tail_std`, `tail_cv`, `drawdown`), 행동/형태(`cap_rate`, `mean_length`, `skew`), 연산(`wall_clock_hours`, `fps`), 종합(`iqm`). 특히 `cap_rate`(평가 에피소드가 타임캡 108,000 스텝에 도달한 비율)는 에이전트가 벽돌을 깨는 것이 아니라 공만 살려두며 시간을 끄는 정책을 학습했는지 여부를 판별하는 핵심 지표이다.

## 5.3 통계 방법: 두 종류의 분산

본 분석은 서로 다른 두 분산을 명확히 구분한다. 첫째, **평가(에피소드) 변동**은 하나의 학습된 모델을 100 에피소드 평가할 때의 변동으로, `summary.json`의 부트스트랩 CI(`final_ci95`)와 Mann-Whitney U 검정, P(A>B), 부트스트랩 평균차 CI가 여기 해당한다. 둘째, **학습(시드) 변동**은 같은 설정을 다른 난수 시드로 재학습했을 때의 변동으로, 시드별 최종 점수를 모아 부트스트랩 95% CI(2,000회)와 IQM 및 IQM의 층화 부트스트랩 CI(5,000회)를 계산하였다. 진정한 다중 시드 스윕(7/77/777)이 존재하는 것은 1M 그룹 전체와 10M의 베이스라인 5종뿐이므로, 알고리즘 비교는 교차 시드로 엄밀히 다루고, 대부분의 ablation(단일 시드 7)은 방향성 있는 단서로만 해석한다.

## 5.4 알고리즘 종합 비교 (10M, 교차 시드)

10M 스텝, 시드 7/77/777 교차 집계 결과는 표 7과 같다. 시각적 비교는 그림 1(교차 시드 막대 + CI + IQM)에 제시하였다.

**표 7**

*10M 스텝 교차 시드 알고리즘 비교 (final_mean 기준)*

| 알고리즘 | 시드 수 | 교차시드 평균 | 95% 부트스트랩 CI | 시드별 점수 (7 / 77 / 777) | 시드 간 std |
|---|---|---|---|---|---|
| A2C | 3 | 207.4 | 113.1 – 386.0 | 386 / 113 / 123 | 126.4 |
| Dueling DQN | 3 | 126.5 | 36.8 – 268.6 | 269 / 74 / 37 | 101.7 |
| PPO | 3 | 102.8 | 22.9 – 261.8 | 262 / 24 / 23 | 112.4 |
| DQN | 3 | 55.0 | 22.1 – 75.7 | 76 / 67 / 22 | 23.5 |
| Double DQN | 3 | 49.8 | 26.6 – 90.4 | 27 / 33 / 90 | 28.7 |

**그림 1**

*교차 시드 알고리즘 성능 비교 (10M 스텝, seeds 7/77/777) — 막대: 평균 ± 95% 부트스트랩 CI, 다이아몬드: IQM, 점: 개별 시드 점수*

![교차 시드 알고리즘 성능 비교](figures/notebook/seed_aggregate.png)

순위는 A2C > Dueling > PPO > DQN ≈ Double DQN이나, CI가 매우 넓어 A2C를 제외하면 막대 간 우열을 단정하기 어렵다. PPO(22.9–261.8), Dueling(36.8–268.6), DQN(22.1–75.7), DDQN(26.6–90.4)의 CI는 크게 겹친다. 시드 7은 A2C·PPO·Dueling에 유리한 실행이었으며, 세 알고리즘 모두 seed 7 점수가 자신의 CI 상단에 위치한다. 6절의 대표값(A2C 386, PPO 262)은 이 운 좋은 단일 시드값이며, 교차 시드 평균은 그 절반 수준이다.

## 5.5 학습 예산 스케일링 (1M → 50M)

표 8은 동일 알고리즘을 예산별로 본 교차 시드 평균이다.

**표 8**

*예산별 교차 시드 평균 성능 (괄호 안은 시드 수)*

| 알고리즘 | 1M | 2M | 10M | 50M |
|---|---|---|---|---|
| A2C | 76.3 (3) | 79.8 (1) | 207.4 (3) | 378.3 (1) |
| PPO | 18.7 (3) | 75.2 (1) | 102.8 (3) | 338.1 (2) |
| DQN | 5.9 (3) | 26.7 (2) | 55.0 (3) | 368.7 (1) |
| Double DQN | 6.7 (3) | — | 49.8 (3) | — |

**그림 2**

*10M 스텝 학습 곡선 — 환경 스텝 대비 평가 보상, 알고리즘별 학습 속도 및 수렴 양상 비교*

![10M 학습 곡선](figures/notebook/learning_curves_10m.png)

50M에서 off-policy DQN(369점)이 A2C(378점)·PPO(338점)에 근접한다. 10M에서 관측된 on-policy의 큰 우위는 영구적 알고리즘 특성이 아니라 짧은 예산의 산물이며, 충분한 스텝이 주어지면 DQN의 샘플 재사용 능력이 on-policy를 따라잡는다는 통설과 부합한다.

예산별 학습 곡선과 최종 평가 점수 분포를 그림 9–14에 제시한다.

**그림 9**

*1M 스텝 학습 곡선 — 초기 학습 단계에서의 알고리즘별 수렴 속도 비교*

![1M 학습 곡선](figures/notebook/learning_curves_1m.png)

**그림 10**

*2M 스텝 학습 곡선*

![2M 학습 곡선](figures/notebook/learning_curves_2m.png)

**그림 11**

*50M 스텝 학습 곡선 — off-policy DQN이 on-policy 알고리즘 성능에 수렴하는 과정*

![50M 학습 곡선](figures/notebook/learning_curves_50m.png)

**그림 12**

*1M 스텝 최종 평가 점수 분포 (평균 ± CI)*

![1M 평가 분포](figures/eval_distribution_1m_2026-06-09_222615.png)

**그림 13**

*10M 스텝 최종 평가 점수 분포*

![10M 평가 분포](figures/eval_distribution_10m_2026-06-09_222621.png)

**그림 14**

*50M 스텝 최종 평가 점수 분포 — DQN과 on-policy 알고리즘의 격차 축소 확인*

![50M 평가 분포](figures/eval_distribution_50m_2026-06-09_222624.png)

## 5.6 샘플 효율·안정성·행동 (seed 7, 10M)

표 9는 다기준 평가 결과이다.

**표 9**

*다기준 평가 결과 (seed 7, 10M 스텝)*

| 알고리즘 | 최종 | AUC | 50점 도달 | cap_rate | 평균 길이 | 왜도 | tail std |
|---|---|---|---|---|---|---|---|
| A2C | 386 | 175.7 | 2.0M | 0.78 | 85,760 | +0.13 | 40.1 |
| Dueling | 269 | 42.6 | 7.0M | 1.00 | 108,000 | −0.54 | 124.1 |
| PPO | 262 | 136.2 | 3.0M | 1.00 | 108,000 | −0.65 | 56.3 |
| DQN | 76 | 23.3 | 6.0M | 0.22 | 28,646 | −1.00 | 7.0 |
| Double DQN | 27 | 17.7 | 미도달 | 1.00 | 108,000 | −0.08 | 1.9 |

**그림 3**

*샘플 효율 비교 — 왼쪽: 학습 곡선 AUC, 오른쪽: 50점 도달 스텝 (미도달 시 10M으로 표시)*

![샘플 효율](figures/notebook/sample_efficiency.png)

**그림 4**

*10M 스텝 최종 평가 보상의 누적 분포 함수(ECDF) — 확률적 지배 관계 시각화*

![ECDF 10M](figures/notebook/ecdf_10m.png)

학습 속도 측면에서 A2C가 가장 빠르고(AUC 176, 2.0M 스텝에 50점 도달) PPO가 뒤따른다(AUC 136, 3.0M). Dueling은 최종 점수는 높지만 AUC가 낮아(43) 늦게 학습하는 유형이다. DDQN(seed 7)은 50점에 끝내 도달하지 못하였다. 행동 측면에서 Double DQN(seed 7)은 `cap_rate=1.00`, 평균 길이 108,000(최댓값)이면서도 점수는 27에 불과하여 벽돌을 깨지 않고 공만 살려 시간을 끄는 정책을 학습하였음을 알 수 있다. 안정성 측면에서 Dueling은 최종 점수가 높지만 `tail_std=124`로 후반 진동이 가장 크다.

## 5.7 연산 비용

10M 스텝 학습 소요 시간은 표 10과 같다.

**표 10**

*알고리즘별 연산 비용 (seed 7, 10M 스텝)*

| 알고리즘 | 학습 시간(h) | FPS |
|---|---|---|
| A2C | 1.49 | 7,570 |
| Double DQN | 1.49 | 7,636 |
| PPO | 1.51 | 7,462 |
| DQN | 1.51 | 7,492 |
| Dueling DQN | 1.53 | 7,366 |

**그림 5**

*연산 비용 대비 성능 — 학습 시간(x축) 대 최종 평가 점수(y축) 산점도, 알고리즘별 라벨*

![연산 비용 대비 성능](figures/notebook/compute_tradeoff.png)

10M 기준 다섯 알고리즘의 벽시계 학습 시간은 약 1.5시간으로 사실상 동일하다(FPS 7.3k–7.8k). 따라서 이 환경·설정에서 알고리즘 선택의 실질적 차별화 요인은 연산 시간이 아니라 동일 시간 내 도달 점수(샘플 효율)이다.

## 5.8 통계적 유의성 (seed 7, 에피소드 수준)

표 11은 가장 큰 공유 예산에서 한 모델씩의 100 평가 에피소드로 계산한 쌍대 비교 결과이다. 이는 단일 모델의 평가 변동에 대한 검정이며 알고리즘 일반 우열의 증거가 아님에 주의해야 한다.

**표 11**

*알고리즘 쌍대 통계 비교 (seed 7, 에피소드 변동 기준)*

| 비교 | 평균차 | 95% 부트스트랩 CI | P(A>B) | Mann-Whitney p |
|---|---|---|---|---|
| A2C > PPO | +124.3 | [104.7, 144.2] | 0.95 | ~1e-28 |
| PPO > DQN | +186.0 | [165.2, 205.8] | 1.00 | ~1e-34 |
| DQN > Double DQN | +49.1 | [43.2, 54.6] | 0.89 | ~1e-21 |
| Dueling vs PPO | +6.9 | [−20.9, 34.9] | 0.52 | 0.55 |

**그림 6**

*쌍대 유의성 히트맵 — 셀 값 = P(행 알고리즘 > 열 알고리즘), 에피소드 변동 기준 (시드 간 비교 아님)*

![유의성 히트맵](figures/notebook/significance.png)

seed 7에서 A2C·PPO·DQN의 분리가 매우 뚜렷한 반면(*p* ≪ .001), Dueling과 PPO는 통계적으로 구분되지 않는다(*P* = .52, *p* = .55).

## 5.9 핵심 관찰: 단일 시드 vs 교차 시드

교차 시드 분석에서 네 가지 핵심 관찰이 도출되었다. 첫째, seed 7은 A2C·PPO·Dueling에 유리한 실행으로, 교차 시드 평균(207/103/127점)이 더 정직한 추정치이다. 둘째, seed 7에서만 보면 DQN(76점) > DDQN(27점)이 결정적으로 보이지만 교차 시드로는 DQN(55점, CI 22–76) ≈ DDQN(50점, CI 27–90)으로 CI가 완전히 겹쳐 차이가 없다. 이는 이론(DDQN ≥ DQN)에 반하는 증거가 아니라 단일 시드 아티팩트였음을 시사한다. 셋째, 교차 시드로 Dueling(127점)은 PPO(103점)를 앞서는 2위이지만 분산이 37–269로 극단적이어서 신뢰하려면 더 많은 시드가 필요하다. 넷째, off-policy의 열세는 10M에 국한되며 50M에서는 사라진다. 따라서 7–8절의 단일 시드 서술은 관측된 현상으로 유효하되, 알고리즘 일반화 주장은 본 절의 교차 시드 결과를 기준으로 읽어야 한다.

---

# 6. 베이스라인 결과

*주의: 본 절의 수치는 단일 시드(seed 7) 결과로 한 모델의 구체적 거동을 보여준다. 시드 분산을 반영한 비교는 5.4절과 5.9절을 함께 참조하기 바란다.*

## 6.1 알고리즘 성능 비교

10M 스텝 학습 후 100 에피소드 평가 결과(seed 7)는 표 12와 같다.

**표 12**

*알고리즘별 베이스라인 성능 (seed 7, 10M 스텝)*

| 알고리즘 | 최종 eval 평균 점수 |
|---|---|
| A2C | 386.03 |
| PPO | 261.77 |
| DQN | 75.73 |
| Double DQN | 26.64 |

**그림 7**

*10M 스텝 알고리즘별 최종 평가 점수 분포 (바이올린 플롯) — 분포 형태·중앙값·사분위수 비교*

![바이올린 플롯 10M](figures/notebook/violin_10m.png)

예산 규모에 따른 분포 변화는 그림 15–17에 제시한다.

**그림 15**

*1M 스텝 알고리즘별 최종 평가 점수 분포 — 학습 초기 단계의 분포 특성*

![바이올린 플롯 1M](figures/notebook/violin_1m.png)

**그림 16**

*2M 스텝 알고리즘별 최종 평가 점수 분포*

![바이올린 플롯 2M](figures/notebook/violin_2m.png)

**그림 17**

*50M 스텝 알고리즘별 최종 평가 점수 분포 — 충분한 예산에서 알고리즘 간 격차 변화*

![바이올린 플롯 50M](figures/notebook/violin_50m.png)

## 6.2 결과 분석

**A2C의 우위.** 10M 스텝이라는 제한된 예산 안에서 A2C가 가장 높은 성능을 보였다. A2C는 매 5 스텝마다 업데이트를 수행하므로 단위 스텝당 업데이트 횟수가 많고, 8개의 병렬 환경에서 수집한 다양한 경험으로 분산을 줄이는 것이 효과적이었던 것으로 보인다.

**PPO의 안정성.** PPO는 A2C보다 낮지만 두 번째로 높은 성능을 기록하였다. Clip 제약으로 인해 정책 업데이트가 안정적이지만, 이 안정성이 초기 학습 속도를 다소 늦추는 단점으로도 작용하였을 수 있다.

**DQN의 낮은 성능.** Off-policy 알고리즘인 DQN은 10M 스텝에서 상대적으로 낮은 점수를 기록하였다. `learning_starts=50,000` 스텝 동안 학습 없이 버퍼를 채우는 초기 단계와 하이퍼파라미터 민감도가 성능에 영향을 미친 것으로 보인다.

**Double DQN의 역설.** 이론적으로는 Double DQN이 표준 DQN보다 과대 추정 편향을 줄여 더 좋은 성능을 보여야 한다 (van Hasselt et al., 2016). 그러나 단일 시드(seed 7)에서는 오히려 낮은 성능(26.64점)을 기록하였다. 이에 대한 이론적 분석은 8.2절에서 다룬다.

---

# 7. Ablation 분석

모든 ablation 실험은 one-knob-at-a-time 원칙에 따라 단 하나의 하이퍼파라미터만 변경하고 나머지는 베이스라인을 유지하였다. 총 10M 스텝, seed 7로 통일하였다. 단일 시드이므로 아래 수치는 방향성 단서로 해석해야 한다 (5.3절 참조).

**그림 8**

*하이퍼파라미터 응답 곡선 전체 스윕 — 각 패널: knob 값(x축) 대비 최종 평가 점수(y축), 베이스라인 점선 표시*

![하이퍼파라미터 응답 곡선](figures/notebook/response_curves.png)

## 7.1 DQN Ablation

### 7.1.1 학습률 (Learning Rate)

**표 13**

*DQN 학습률 ablation 결과*

| 설정 | `learning_rate` | 최종 평균 점수 |
|---|---|---|
| 베이스라인 | 1.0e-4 | 75.73 |
| `dqn_lr_1.5e-4` | 1.5e-4 | 191.17 |

학습률을 1.5배 높였을 때 성능이 2.5배 이상 향상되는 극적인 결과를 얻었다. 1.0e-4는 SB3의 보수적 기본값으로 Breakout의 보상 밀도를 감안하면 다소 느린 학습 속도를 유발하였을 수 있다. Adam 옵티마이저는 적응적 학습률을 내부적으로 계산하므로 글로벌 학습률이 최종 업데이트 크기에 미치는 영향이 SGD보다 간접적이다. 그러나 베이스라인의 1.0e-4가 충분히 탐색적이지 않았음을 이 결과가 시사하며, 학습률이 ε-greedy 탐색과 더불어 DQN 성능에 가장 민감하게 작용하는 하이퍼파라미터임을 확인할 수 있다.

### 7.1.2 Target Network 업데이트 주기

**표 14**

*DQN Target Network 업데이트 주기 ablation 결과*

| 설정 | `target_update_interval` | 최종 평균 점수 |
|---|---|---|
| 베이스라인 | 1,000 steps | 75.73 |
| `dqn_target_5000` | 5,000 steps | 163.30 |

업데이트 주기를 5배 늘렸을 때 163.3점으로 크게 향상되었다. Target network의 역할은 Q-learning 타깃을 안정적으로 유지하는 것이다. 업데이트 주기가 짧을수록(1,000) 타깃이 자주 변하여 학습이 불안정해진다. Mnih et al. (2015)의 원 논문에서는 10,000 스텝마다 업데이트하였으나, 베이스라인의 1,000은 지나치게 잦은 업데이트로 타깃의 안정성 효과가 반감되고 있었음을 보여준다.

### 7.1.3 Replay Buffer 크기

**표 15**

*DQN Replay Buffer 크기 ablation 결과*

| 설정 | `buffer_size` | 최종 평균 점수 |
|---|---|---|
| `dqn_buffer_100k` | 100,000 | 47.00 |
| 베이스라인 | 250,000 | 75.73 |
| `dqn_buffer_300k` | 300,000 | 8.70 |

버퍼를 줄이면 성능이 하락하고, 늘리면 오히려 더 크게 하락하는 역설적 결과가 나타났다. Buffer 100k의 하락은 최근 경험에 편중되어 샘플 다양성이 줄어들고 분포 불일치(distribution shift)가 심해지기 때문이다. Buffer 300k의 역설적 하락은 10M 스텝이라는 제한된 학습 예산 안에서 버퍼가 클수록 현재 정책과 동떨어진 과거 경험이 더 오랫동안 잔류하여 off-policy 편향이 증가하기 때문으로 해석된다.

### 7.1.4 Frame Stacking

**표 16**

*DQN Frame Stacking ablation 결과*

| 설정 | `frame_stack` | 최종 평균 점수 |
|---|---|---|
| `dqn_fs_1` | 1 | 4.66 ~ 9.11 |
| 베이스라인 | 4 | 75.73 |
| `dqn_fs_6` | 6 | 45.60 |

Frame stacking을 제거하면 에이전트가 사실상 작동하지 않고(≈5점), 4→6으로 늘리면 소폭 하락한다. Breakout에서 공의 속도와 방향은 단일 프레임에서 알 수 없어 Partial Observability 문제가 발생한다 (Mnih et al., 2015). Frame stacking은 여러 프레임을 쌓아 속도와 방향 정보를 암묵적으로 인코딩하며, fs=1에서 거의 작동하지 않는 것은 이 Markov property 위반이 얼마나 치명적인지를 직접 증명한다.

## 7.2 PPO Ablation

### 7.2.1 Clip Range

**표 17**

*PPO Clip Range ablation 결과*

| 설정 | `clip_range` | 최종 평균 점수 |
|---|---|---|
| `ppo_clip_0.05` | 0.05 | 270.55 |
| 베이스라인 | 0.1 | 261.77 |
| `ppo_clip_0.2` | 0.2 | 263.76 |
| `ppo_clip_0.3` | 0.3 | 361.71 |

clip_range를 크게 늘릴수록 성능이 향상되는 추세가 명확하며, 0.3에서 베이스라인 대비 38% 향상이 나타났다. Atari 게임처럼 희소한 보상(sparse reward) 환경에서는 좋은 경험이 드물게 발생하므로 그 경험으로부터 최대한 많이 배우는 것이 유리하다. 다만 clip_range를 무한정 늘리면 trust region 보장이 사라지고 분산이 증가할 수 있어, 이 데이터만으로 "클수록 좋다"고 결론 짓는 것은 위험하다.

### 7.2.2 GAE Lambda

**표 18**

*PPO GAE Lambda ablation 결과*

| 설정 | `gae_lambda` | 최종 평균 점수 |
|---|---|---|
| `ppo_gae_0.9` | 0.90 | 331.21 |
| 베이스라인 | 0.95 | 261.77 |
| `ppo_gae_1.0` | 1.0 | 48.99 |

λ=1.0(Monte-Carlo advantage)이 급격히 낮은 성능을 보였다. GAE (Schulman et al., 2016)에서 λ는 bias-variance tradeoff를 제어한다. λ=0은 TD(0) advantage(높은 편향, 낮은 분산)를, λ=1은 Monte-Carlo advantage(낮은 편향, 높은 분산)를 의미한다. Breakout은 에피소드 길이가 수천 스텝에 달할 수 있어 λ=1의 높은 분산이 그래디언트 추정치를 불안정하게 만드는 것으로 해석된다. λ=0.9가 베이스라인 0.95보다 더 높은 성능을 보인 것은 분산 감소의 이점이 이 환경에서 크다는 것을 시사한다.

### 7.2.3 N_epochs

**표 19**

*PPO n_epochs ablation 결과*

| 설정 | `n_epochs` | 최종 평균 점수 |
|---|---|---|
| 베이스라인 | 4 | 261.77 |
| `ppo_ne_10` | 10 | 61.65 |

n_epochs를 4→10으로 늘렸을 때 성능이 약 76% 하락하였다. PPO는 On-policy 알고리즘이므로 수집된 경험은 현재 정책으로 만들어진 것이다. 같은 데이터를 여러 번 재사용(10 epochs)하면 후반 업데이트에서 현재 정책과 데이터 수집 정책이 너무 달라져 importance ratio $r_t(\theta)$가 Clip 범위를 벗어나는 policy drift 현상이 발생하여 학습이 불안정해진다.

### 7.2.4 Entropy 계수

**표 20**

*PPO Entropy 계수 ablation 결과*

| 설정 | `ent_coef` | 최종 평균 점수 |
|---|---|---|
| `ppo_ent_0.0` | 0.0 | 272.94 |
| 베이스라인 | 0.01 | 261.77 |
| `ppo_ent_0.05` | 0.05 | 346.14 |

엔트로피 보너스를 높이면 성능이 향상되고, 제거하면 비슷하거나 약간 향상된다. 엔트로피 보너스는 정책이 확률 분포를 균일하게 유지하도록 장려하여 과조기 수렴(premature convergence)을 방지한다. ent_coef=0.05에서의 향상은 Breakout처럼 다양한 전략이 존재하는 환경에서 더 많은 탐색이 유리함을 시사한다.

### 7.2.5 기타 PPO Ablation 결과 요약

**표 21**

*기타 PPO Ablation 결과*

| 설정 | 변경 내용 | 최종 평균 점수 | 해석 |
|---|---|---|---|
| `ppo_gamma_0.995` | γ 0.99 → 0.995 | 231.55 | 더 긴 시야가 오히려 학습 불안정 |
| `ppo_lr_1e-4` | lr 2.5e-4 → 1e-4 | 245.54 | 학습 속도 저하 |

## 7.3 A2C Ablation

### 7.3.1 Advantage 정규화 (Normalize Advantage)

**표 22**

*A2C Advantage 정규화 ablation 결과*

| 설정 | `normalize_advantage` | 최종 평균 점수 |
|---|---|---|
| 베이스라인 | False | 386.03 |
| `a2c_normadv` | True | 10.54 |

Advantage 정규화를 활성화했을 때 성능이 386.03 → 10.54로 97% 폭락하였다. 이는 본 실험의 가장 충격적인 결과 중 하나이다. Advantage 정규화는 미니배치 내의 advantage를 평균 0, 분산 1로 표준화하는 기법으로 PPO에서는 안정적인 학습에 기여한다고 알려져 있다. 그러나 A2C에서는 치명적인 결과를 낳았다. A2C의 `n_steps=5`는 8개의 환경 × 5 스텝 = 40개의 전이만으로 advantage를 계산한다. 40개 샘플에서 추정한 평균과 표준편차는 통계적으로 매우 불안정하며, 특히 에피소드 시작 직후처럼 advantage 값이 대부분 비슷한 상황에서는 표준편차가 0에 가까워져 수치 폭발이 발생할 수 있다. 반면 PPO는 `n_steps=128 × 8 env = 1,024`개의 advantage를 모아서 정규화하므로 통계적으로 훨씬 안정적이다. 즉, advantage 정규화는 충분한 배치 크기가 확보되어야 효과적이며, A2C의 짧은 rollout에서는 오히려 그래디언트 방향을 왜곡하는 부작용을 낳는다.

### 7.3.2 Value Function 계수 (vf_coef)

**표 23**

*A2C Value Function 계수 ablation 결과*

| 설정 | `vf_coef` | 최종 평균 점수 |
|---|---|---|
| 베이스라인 | 0.25 | 386.03 |
| `a2c_vf_0.5` | 0.5 | 356.50 |

vf_coef를 PPO 기본값(0.5)으로 높여도 성능이 소폭 하락할 뿐 크게 유지된다. vf_coef는 Actor-Critic 손실 함수에서 Critic 업데이트의 가중치를 결정한다. A2C의 기본값 0.25는 가치 학습보다 정책 학습에 더 집중하는 설정이며, 결과적으로 A2C에서 vf_coef가 비교적 덜 민감한 하이퍼파라미터임을 시사한다.

### 7.3.3 옵티마이저 (Optimizer)

**표 24**

*A2C 옵티마이저 ablation 결과*

| 설정 | 옵티마이저 | 최종 평균 점수 |
|---|---|---|
| 베이스라인 | RMSProp | 386.03 |
| `a2c_adam` | Adam | 305.84 |

Adam으로 교체하면 성능이 약 21% 하락한다. RMSProp은 A3C 원 논문 (Mnih et al., 2016)에서 비동기 또는 단기 rollout 환경에 적합한 옵티마이저로 선택되었다. 비정상적(non-stationary) 그래디언트 환경에서 최근 그래디언트 제곱의 이동 평균으로 학습률을 조절하는 방식이 이 설정에 유리하다. 성능 하락은 학습률 재튜닝 없이 옵티마이저만 교체했기 때문일 가능성이 높다.

### 7.3.4 기타 A2C Ablation 결과 요약

**표 25**

*기타 A2C Ablation 설계 (미완료)*

| 설정 | 변경 내용 | 상태 |
|---|---|---|
| `a2c_lr_2.5e-4` | lr 7e-4 → 2.5e-4 | 미완료 |
| `a2c_nstep_16` | n_steps 5 → 16 | 미완료 |
| `a2c_gae_0.95` | gae_lambda 1.0 → 0.95 | 미완료 |
| `a2c_nstep16_gae095` | n_steps=16 + gae_lambda=0.95 | 미완료 |

특히 `a2c_nstep16_gae095`는 A2C를 PPO와 유사한 어드밴티지 추정 방식으로 바꾸는 흥미로운 실험으로, 향후 과제로 남긴다.

---

# 8. 종합 분석 및 이론적 반성

## 8.1 On-policy vs Off-policy: Breakout에서 A2C의 우위

10M 스텝 예산에서 On-policy 알고리즘(A2C, PPO)이 Off-policy 알고리즘(DQN)을 크게 앞서는 결과는 직관에 반할 수 있다. 일반적으로 Off-policy 알고리즘은 Replay Buffer를 통한 샘플 재사용으로 샘플 효율이 높다고 알려져 있기 때문이다. 그러나 다음 세 요소가 이 결과를 설명한다.

첫째, **병렬 환경의 영향**이다. A2C와 PPO는 8개의 환경을 병렬로 실행하므로 실제로는 10M 스텝이지만 다양한 게임 상태를 8배 더 효율적으로 탐색한다. 반면 DQN은 1개의 환경만 사용한다.

둘째, **업데이트 빈도**의 차이이다. A2C는 매 5×8=40 전이마다 업데이트한다. DQN은 4 스텝마다 1회 업데이트하지만 처음 50,000 스텝은 업데이트 없이 버퍼를 채우므로 실제 그래디언트 업데이트 횟수 차이가 크다.

셋째, **Breakout의 보상 구조**이다. Breakout은 벽돌을 맞출 때마다 즉각적인 보상이 발생하는 비교적 밀집된 보상(dense reward) 구조이다. 이런 환경에서는 On-policy 알고리즘도 학습 신호를 충분히 얻을 수 있어 Off-policy의 샘플 효율 우위가 줄어든다.

## 8.2 Double DQN이 DQN보다 낮은 이유

단일 시드(seed 7)에서 관측된 Double DQN(26.64점) < DQN(75.73점)의 역설은 교차 시드로는 사라진다(5.9절). 단일 시드 관측에 대한 가능한 해석은 세 가지이다. 첫째, **탐색 관점**에서 표준 DQN의 과대 추정 편향이 오히려 초기 탐색에 유리하게 작용하였을 가능성이 있다. 둘째, **단일 시드 한계**로 Double DQN의 성능이 시드에 따른 분산이 큰 경우에 해당할 수 있다. 셋째, **하이퍼파라미터 의존성**으로 Double DQN의 이점은 target_update_interval이 충분히 클 때 더 두드러질 수 있는데, 본 실험의 1,000이 지나치게 잦아서 Double DQN의 안정화 효과가 표준 DQN의 탐색적 편향을 충분히 상쇄하지 못하였을 가능성이 있다.

## 8.3 하이퍼파라미터 민감도 비교

표 26은 실험 전반을 통해 확인된 알고리즘별 하이퍼파라미터 민감도 수준을 요약한다.

**표 26**

*알고리즘별 하이퍼파라미터 민감도 비교*

| 알고리즘 | 민감도 | 주요 민감 파라미터 |
|---|---|---|
| DQN | 매우 높음 | lr, target_update_interval, buffer_size |
| PPO | 중간 | clip_range, gae_lambda, n_epochs |
| A2C | 중간 (특이점 존재) | normalize_advantage (파국적 실패), optimizer |

DQN은 하이퍼파라미터 변경에 극히 민감하여 lr 변화 하나로 2.5배 이상의 성능 차이를 보였다. Off-policy 학습 특성상 Replay Buffer의 데이터 분포와 타깃 네트워크의 안정성이 복잡하게 상호작용하기 때문이다. PPO는 n_epochs=10처럼 명백히 잘못된 설정을 제외하면 상대적으로 견고하다. A2C는 이론적으로 무해해 보이는 normalize_advantage가 파국적 결과를 낳을 수 있음을 보여주었으며, 이는 짧은 rollout(n_steps=5)으로 인한 통계적 불안정성이 숨겨진 위험 요소임을 시사한다.

## 8.4 한계와 신뢰성

본 실험의 세 가지 한계점을 명시한다. 첫째, ablation 실험 전반이 seed=7 단일 실행이다. RL 학습은 시드에 따른 분산이 매우 크므로 일부 결과(특히 DDQN의 낮은 성능)는 재현성 문제일 수 있다. 둘째, 10M 스텝의 제약으로 DQN 계열은 더 긴 학습(50M+ 스텝)에서 On-policy 알고리즘을 따라잡는다. 본 결과는 짧은 예산에서의 특수 케이스이다. 셋째, 튜닝된 베이스라인의 부재로, DQN의 최적 하이퍼파라미터(lr=1.5e-4, target=5,000 조합)라면 알고리즘 순위가 달라질 수 있다.

---

# 9. 결론

## 9.1 주요 발견 요약

다섯 가지 핵심 발견을 요약한다. 첫째, **알고리즘 순위**는 10M·seed 7 기준 A2C(386점) > PPO(262점) >> DQN(76점) > DDQN(27점)이나, 교차 시드 순위는 A2C(207점) > Dueling(127점) > PPO(103점) > DQN(55점) ≈ DDQN(50점)으로 달라진다. 둘째, **가장 극적인 ablation 결과**는 DQN lr 1.5e-4에서 75.73 → 191.17(2.5× 향상)과 A2C normalize_advantage에서 386 → 10(97% 하락)이다. 셋째, **이론과 실험의 불일치**로 Double DQN의 이론적 우위가 단일 시드 10M 스텝에서는 관찰되지 않았으나 이는 시드 아티팩트임이 교차 시드 분석에서 확인되었다. 넷째, **A2C normalize_advantage 실패**는 배치 크기(40개)가 너무 작을 때 advantage 정규화가 도움이 아닌 해가 됨을 보여준다. 다섯째, **DQN의 하이퍼파라미터 민감도**로 Target network 업데이트 주기와 learning rate가 DQN 성능에 가장 큰 영향을 미친다.

## 9.2 실무적 시사점

제한된 예산 환경에서는 On-policy 알고리즘(특히 A2C, PPO)이 Off-policy보다 구현과 튜닝이 쉽고 안정적이다. DQN 사용 시 target_update_interval을 5,000~10,000으로 유지하고 learning rate를 신중하게 선택하는 것이 중요하다. PPO 튜닝 시 clip_range를 늘리는 것이 Atari 환경에서 효과적일 수 있으며, n_epochs를 4 이상으로 높이는 것은 위험하다. A2C 사용 시 normalize_advantage는 반드시 해제해야 하며 optimizer 변경 시 learning rate 재튜닝이 필요하다.

## 9.3 향후 방향

향후 연구 방향으로 다수 시드 반복 실험을 통한 신뢰 구간 기반의 통계적으로 유의한 비교, DQN 최적 설정 조합(lr=1.5e-4 + target_update=5,000)의 성능 확인, PPO clip_range=0.3 이상의 영역에서 성능 포화점 확인, 그리고 Prioritized Experience Replay 및 n-step return 등을 추가한 Rainbow DQN (Hessel et al., 2018)으로의 확장을 제안한다.

---

# 참고문헌

Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., & Silver, D. (2018). Rainbow: Combining improvements in deep reinforcement learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, *32*(1).

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning*, *48*, 1928–1937.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, *518*(7540), 529–533. https://doi.org/10.1038/nature14236

Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *Proceedings of the 4th International Conference on Learning Representations (ICLR)*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms*. arXiv. https://arxiv.org/abs/1707.06347

van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *Proceedings of the 30th AAAI Conference on Artificial Intelligence*, 2094–2100.

Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning*, *48*, 1995–2003.

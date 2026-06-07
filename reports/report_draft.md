# ALE/Breakout-v5 강화학습 실험 보고서

**제출일:** 2026년 6월 10일  
**환경:** ALE/Breakout-v5  
**라이브러리:** Stable-Baselines3 (커스텀 확장 포함)

---

## 목차

1. [서론](#1-서론)
2. [환경 및 전처리](#2-환경-및-전처리)
3. [알고리즘 구현](#3-알고리즘-구현)
4. [실험 설정](#4-실험-설정)
5. [베이스라인 결과](#5-베이스라인-결과)
6. [Ablation 분석](#6-ablation-분석)
7. [종합 분석 및 이론적 반성](#7-종합-분석-및-이론적-반성)
8. [결론](#8-결론)

---

## 1. 서론

Atari 게임 Breakout은 강화학습(Reinforcement Learning, RL) 연구의 표준 벤치마크 환경 중 하나다. 에이전트는 픽셀 이미지만을 입력으로 받아 공을 벽돌에 맞히는 전략을 스스로 학습해야 하므로, 시각적 인식, 장기 보상 할인, 탐색-이용 균형 등 RL의 핵심 문제들이 모두 등장한다.

본 프로젝트의 목표는 단순히 높은 점수를 달성하는 것이 아니라, **다양한 알고리즘과 하이퍼파라미터 실험을 통해 RL 이론과 실전 결과의 차이를 분석**하는 것이다.

### 1.1 라이브러리 활용 방식

본 프로젝트는 Stable-Baselines3(SB3)를 기반으로 하되, 단순 실행에 머물지 않고 다음과 같은 **원본 기여(original contributions)**를 추가하였다.

| 알고리즘 | SB3 활용 방식 |
|---|---|
| DQN | SB3 `DQN` 클래스 그대로 사용 |
| Double DQN | SB3 `DQN`을 **상속**하여 `train()` 메서드 오버라이드 — Double Q-learning 타깃 직접 구현 |
| Dueling DQN | SB3 `QNetwork`를 **상속**하여 Value/Advantage 분기 네트워크로 교체, `DuelingCnnPolicy` 정의 |
| PPO | SB3 `PPO` 래퍼 — YAML 설정에서 하이퍼파라미터 주입 |
| A2C | SB3 `A2C` 래퍼 — YAML 설정에서 하이퍼파라미터 주입 |

즉, PPO와 A2C는 SB3의 검증된 구현을 그대로 활용하되 하이퍼파라미터의 선택 근거를 명시하였고, DQN 계열은 핵심 수식(Double Q-learning target, Dueling head)을 직접 구현하여 이론적 이해를 코드로 표현하였다.

---

## 2. 환경 및 전처리

### 2.1 환경 설정

- **환경 ID:** `ALE/Breakout-v5`
- **패키지:** `gymnasium[atari]==1.3.0`, `ale-py==0.11.2`, `autorom[accept-rom-license]==0.6.1`
- **ALE 설정 (Sticky action 제거):**
  - `frameskip=1` — SB3의 `MaxAndSkip` 래퍼가 frame skip을 직접 처리
  - `repeat_action_probability=0.0` — 무작위 행동 반복 비활성화 (공정한 비교를 위해)
  - `full_action_space=False` — Breakout의 필수 행동(4개)만 사용

### 2.2 전처리 파이프라인

SB3의 `make_atari_env()`는 내부적으로 `AtariWrapper`를 적용하며, 다음의 전처리 단계로 구성된다.

```
원본 RGB 프레임 (210×160×3)
    ↓ MaxAndSkip(4)     : 4 프레임마다 최대 픽셀을 취해 에이전트에게 전달
    ↓ NoopReset         : 에피소드 시작 시 0-30회 무작위 NOP 실행 (초기 상태 다양화)
    ↓ EpisodicLife      : 생명 소모 시 에피소드 종료 처리 (학습 시에만)
    ↓ FireReset         : FIRE 버튼이 필요한 게임에서 에피소드 시작 처리
    ↓ WarpFrame(84×84)  : 그레이스케일 변환 + 84×84 리사이즈
    ↓ ClipReward        : 보상을 [-1, +1]로 클리핑 (학습 안정성)
    ↓ VecFrameStack(4)  : 연속 4 프레임 스택 (속도·방향 정보 제공)
최종 입력 형태: (4, 84, 84)
```

### 2.3 학습 환경과 평가 환경의 차이

| 항목 | 학습 환경 | 평가 환경 |
|---|---|---|
| `terminal_on_life_loss` | `True` (생명 소모 = 에피소드 종료) | `False` (게임 전체가 1 에피소드) |
| 보상 클리핑 | `[-1, +1]` | 없음 (실제 점수) |
| 병렬 env 수 | 8 | 1 |

학습 시 생명 소모를 에피소드 종료로 처리하는 것은 에이전트가 "죽지 않는 것"이 즉각적인 부정적 결과임을 더 빠르게 학습하게 하는 일반적인 Atari 학습 기법이다. 그러나 평가는 **실제 게임 완주 점수**로 측정하므로, 두 환경을 분리하였다.

---

## 3. 알고리즘 구현

### 3.1 DQN (Deep Q-Network)

**원본 논문:** Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015

DQN은 Q-함수를 신경망으로 근사하고, Experience Replay와 Target Network로 학습을 안정화한다.

**Q-learning 업데이트 타깃 (표준 DQN):**

$$y_i = r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a')$$

핵심 안정화 기법:
- **Experience Replay**: 과거 전이 $(s, a, r, s')$를 Replay Buffer에 저장하고 미니배치로 샘플링하여 상관관계를 끊음
- **Target Network**: Q-learning 타깃 계산에 주기적으로 복사된 별도 네트워크를 사용하여 타깃 변동을 줄임

### 3.2 Double DQN

**원본 논문:** van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI 2016

> **본 프로젝트의 원본 기여:** Double DQN의 학습 루프(`train()` 메서드)를 SB3 내부를 수정하지 않고, `DoubleDQN(DQN)` 서브클래스로 직접 구현하였다. SB3의 표준 DQN에는 이 로직이 존재하지 않는다.

표준 DQN은 $\max_{a'} Q(s', a')$를 타깃 네트워크 하나로 계산하므로 최대치를 과대 추정하는 경향이 있다. Double DQN은 **행동 선택**과 **Q값 평가**를 분리하여 이 bias를 줄인다.

**Double DQN 타깃:**

$$y_i = r + \gamma \cdot Q_{\text{target}}\!\left(s',\; \arg\max_{a'} Q_{\text{online}}(s', a')\right)$$

SB3의 `DQN`을 상속하여 `train()` 메서드를 오버라이드하였다. SB3 원본의 `max_a' Q_target(s', a')`를 아래 두 단계로 교체한 것이 전부이자, Double DQN의 핵심이다.

```python
# [원본 기여] 표준 DQN과 다른 부분: 행동 선택을 온라인 네트워크가 담당
next_q_online = self.q_net(replay_data.next_observations)
next_actions = next_q_online.argmax(dim=1, keepdim=True)
# Q-value 평가는 타깃 네트워크가 담당 → 과대 추정 bias 감소
next_q_target = self.q_net_target(replay_data.next_observations)
next_q_values = next_q_target.gather(1, next_actions)
```

### 3.3 Dueling DQN

**원본 논문:** Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016

> **본 프로젝트의 원본 기여 (아키텍처 수정):** SB3에는 Dueling 네트워크가 내장되어 있지 않다. 본 프로젝트에서는 SB3의 `QNetwork`를 직접 상속하여 Dueling head를 구현하고, 이를 SB3의 정책 시스템과 연결하는 `DuelingCnnPolicy`를 새로 정의하였다.

Dueling DQN은 Q-함수를 **상태 가치 V(s)** 와 **행동 어드밴티지 A(s,a)** 의 합으로 분해한다.

$$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)$$

평균을 빼는 것은 V(s)와 A(s,a)의 **정체성 모호성(identifiability)**을 해소하기 위해서다. 즉, 같은 Q값이 V와 A를 무한히 다양하게 조합으로 표현될 수 있는 문제를 방지한다.

**[아키텍처 수정]** SB3의 `QNetwork`를 상속하여, 기존의 단일 FC 출력 레이어(`q_net`)를 `value_net`(1차원 출력)과 `advantage_net`(n_actions 차원 출력)으로 분리·교체하였다.

```
NatureCNN (공유 특징 추출기, SB3 기본)
    │
    ├─► value_net:     FC(512) → Linear(1)          → V(s)
    └─► advantage_net: FC(512) → Linear(n_actions)  → A(s,a)
                                                        │
                              Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
```

```python
# [원본 구현] DuelingQNetwork.forward()
def forward(self, obs):
    features = self.extract_features(obs, self.features_extractor)
    value = self.value_net(features)            # V(s): shape (B, 1)
    advantage = self.advantage_net(features)    # A(s,a): shape (B, n_actions)
    return value + advantage - advantage.mean(dim=1, keepdim=True)
```

이 `DuelingQNetwork`를 SB3 정책 시스템에 연결하기 위해 `DuelingCnnPolicy`를 정의하고, `make_q_net()` 메서드를 오버라이드하여 빌더에서 `dueling=True` 플래그 시 자동으로 선택되도록 구성하였다.

### 3.4 PPO (Proximal Policy Optimization)

**원본 논문:** Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017

PPO는 On-policy 정책 그래디언트 알고리즘으로, 정책 업데이트가 너무 크게 변하지 않도록 **Clip** 또는 **KL 패널티**로 제약을 건다. 본 실험에서는 Clip 방식을 사용하였다.

**PPO-Clip 목적 함수:**

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

여기서 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$는 정책 비율, $\hat{A}_t$는 GAE로 추정한 어드밴티지이다.

SB3의 PPO 구현을 그대로 사용하되, YAML 설정 파일에서 모든 하이퍼파라미터를 주입한다.

### 3.5 A2C (Advantage Actor-Critic)

**원본 논문:** Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016 (A3C의 동기 버전)

A2C는 Actor(정책)와 Critic(가치 함수)을 동시에 학습하는 On-policy 알고리즘이다. A3C와 달리 여러 환경을 동기적으로 실행하여 경험을 수집한다.

**A2C 업데이트:**

$$\nabla_\theta J(\theta) \approx \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]$$

$$\hat{A}_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)$$

SB3 기본 A2C는 옵티마이저로 **RMSProp**을 사용한다. 이는 A3C 원 논문의 설정을 따른 것으로, 비정상적(non-stationary) 그래디언트 환경에서 적응적 학습률을 제공한다.

---

## 4. 실험 설정

### 4.1 공통 설정

| 항목 | 값 |
|---|---|
| 총 학습 스텝 | 10,000,000 |
| 시드 | 7 |
| 특징 추출기 | NatureCNN (Mnih 2015) |
| 학습용 병렬 env | 8 |
| 평가용 env | 1 |
| 평가 에피소드 수 | 100 |

### 4.2 알고리즘별 하이퍼파라미터

**DQN / Double DQN / Dueling DQN:**

| 하이퍼파라미터 | 값 | 근거 |
|---|---|---|
| `learning_rate` | 1.0e-4 | SB3 Atari DQN 기본값 |
| `buffer_size` | 250,000 | VRAM/RAM 제약 하의 절충 (Mnih 2015의 1M 대비 축소) |
| `batch_size` | 32 | Nature DQN 원 논문 값 |
| `train_freq` | 4 steps | 4 스텝마다 1회 업데이트 |
| `target_update_interval` | 1,000 | SB3 기본값 (Mnih 2015의 10,000 대비 공격적) |
| `exploration_fraction` | 0.1 | 전체의 10%에서 ε 1.0 → 0.01로 감소 |
| `gamma` | 0.99 | 표준값 |
| `max_grad_norm` | 10.0 | — |

**PPO:**

| 하이퍼파라미터 | 값 | 근거 |
|---|---|---|
| `learning_rate` | 2.5e-4 | Schulman 2017 Atari PPO 표준 |
| `n_steps` | 128 | 128 × 8 env = 1,024 steps/update |
| `batch_size` | 256 | — |
| `n_epochs` | 4 | SB3 Atari 기본값 |
| `gamma` | 0.99 | 표준값 |
| `gae_lambda` | 0.95 | GAE 원 논문 (Schulman 2016) 기본값 |
| `clip_range` | 0.1 | SB3 Atari 기본값 (원 논문 0.2보다 보수적) |
| `ent_coef` | 0.01 | 탐색 장려를 위한 엔트로피 보너스 |
| `vf_coef` | 0.5 | 가치 손실 가중치 |
| `max_grad_norm` | 0.5 | — |

**A2C:**

| 하이퍼파라미터 | 값 | 근거 |
|---|---|---|
| `learning_rate` | 7.0e-4 | A3C/A2C 표준 RMSProp LR |
| `n_steps` | 5 | 단기 n-step return (원 논문 방식) |
| `gamma` | 0.99 | 표준값 |
| `gae_lambda` | 1.0 | λ=1 = 순수 n-step return (GAE 미사용) |
| `ent_coef` | 0.01 | 엔트로피 보너스 |
| `vf_coef` | 0.25 | SB3 A2C Atari 기본값 |
| `use_rms_prop` | True | A3C 원 논문의 RMSProp 사용 |
| `normalize_advantage` | False | 기본값 |
| `rms_prop_eps` | 1.0e-5 | — |

---

## 5. 베이스라인 결과

### 5.1 알고리즘 성능 비교

10M 스텝 학습 후 100 에피소드 평가 결과 (seed 7):

| 알고리즘 | 최종 eval 평균 점수 |
|---|---|
| **A2C** | **386.03** |
| PPO | 261.77 |
| DQN | 75.73 |
| Double DQN | 33.10 |

![베이스라인 학습 곡선](figures/learning_curves_10m.png)

![베이스라인 평가 분포](figures/eval_distribution_10m.png)

### 5.2 결과 분석

**A2C의 우위:** 10M 스텝이라는 제한된 예산 안에서 A2C가 가장 높은 성능을 보였다. A2C는 매 5 스텝마다 업데이트를 수행하므로 단위 스텝당 업데이트 횟수가 많고, 8개의 병렬 환경에서 수집한 다양한 경험으로 분산을 줄이는 것이 효과적이었던 것으로 보인다.

**PPO의 안정성:** PPO는 A2C보다 낮지만 두 번째로 높은 성능을 기록했다. Clip 제약으로 인해 정책 업데이트가 안정적이지만, 이 안정성이 초기 학습 속도를 다소 늦추는 단점으로도 작용했을 수 있다.

**DQN의 낮은 성능:** Off-policy 알고리즘인 DQN은 10M 스텝에서 상대적으로 낮은 점수를 기록했다. DQN은 Replay Buffer에서 과거 경험을 재사용하므로 샘플 효율이 높을 것으로 예상되지만, 실제로는 탐색 초기 단계(learning_starts=50,000 스텝 동안 학습 없음)와 하이퍼파라미터 민감도가 성능에 영향을 미친 것으로 보인다.

**Double DQN의 역설:** 이론적으로는 Double DQN이 표준 DQN보다 과대 추정 bias를 줄여 더 좋은 성능을 보여야 한다. 그러나 본 실험에서는 오히려 낮은 성능(33.10)을 기록하였다. 이에 대한 이론적 분석은 7.2절에서 다룬다.

---

## 6. Ablation 분석

모든 ablation 실험은 **one-knob-at-a-time** 원칙에 따라 단 하나의 하이퍼파라미터만 변경하고, 나머지는 베이스라인을 그대로 유지하였다. 총 10M 스텝, seed 7로 통일하였다.

### 6.1 DQN Ablation

#### 6.1.1 학습률 (Learning Rate)

| 설정 | `learning_rate` | 최종 평균 점수 |
|---|---|---|
| 베이스라인 | 1.0e-4 | 75.73 |
| `dqn_lr_1.5e-4` | **1.5e-4** | **191.17** |
| `dqn_lr_2.5e-4` | 2.5e-4 | (미완료) |

**결과:** lr을 1.5배 높였을 때 성능이 2.5배 이상 향상되는 극적인 결과를 얻었다. 1.0e-4는 SB3의 보수적 기본값으로, Breakout의 보상 밀도를 감안하면 다소 느린 학습속도를 유발했을 수 있다. 반면, Mnih 2015 원 논문에서 RMSProp으로 사용하던 2.5e-4는 Adam과의 조합에서 지나치게 공격적일 수 있으므로 1.5e-4가 최적에 가까운 것으로 보인다.

**이론적 해석:** Adam 옵티마이저는 적응적 학습률을 내부적으로 계산하므로, 글로벌 학습률이 최종 업데이트 크기에 미치는 영향이 SGD보다 간접적이다. 그러나 베이스라인의 1.0e-4가 충분히 탐색적이지 않았음을 시사하며, 학습률이 ε-greedy 탐색과 더불어 DQN의 성능에 가장 민감하게 작용하는 하이퍼파라미터임을 확인할 수 있다.

#### 6.1.2 Target Network 업데이트 주기

| 설정 | `target_update_interval` | 최종 평균 점수 |
|---|---|---|
| 베이스라인 | 1,000 steps | 75.73 |
| `dqn_target_5000` | **5,000 steps** | **163.30** |
| `dqn_target_10000` | 10,000 steps | (미완료) |

**결과:** Target network 업데이트 주기를 5배 늘렸을 때 163.3으로 크게 향상되었다.

**이론적 해석:** Target network의 역할은 Q-learning 타깃을 안정적으로 유지하는 것이다. 업데이트 주기가 짧을수록(1,000) 타깃이 자주 변하여 학습이 불안정해진다. 반면 Mnih 2015 원 논문에서는 10,000 스텝마다 업데이트하였다. 베이스라인의 1,000은 지나치게 잦은 업데이트로 인해 타깃의 안정성 효과가 반감되고 있었음을 이 결과가 보여준다.

#### 6.1.3 Replay Buffer 크기

| 설정 | `buffer_size` | 최종 평균 점수 |
|---|---|---|
| 베이스라인 | 250,000 | 75.73 |
| `dqn_buffer_100k` | **100,000** | 47.00 |
| `dqn_buffer_300k` | **300,000** | **8.70** |

**결과:** Buffer를 줄이면 성능이 하락하고, 늘리면 오히려 더 크게 하락하는 역설적 결과가 나타났다.

**이론적 해석:**
- **Buffer 100k 하락:** 작은 버퍼는 최근 경험에 편중되어 샘플 다양성이 줄어들고, Off-policy 학습에서 분포 불일치(distribution shift)가 심해진다.
- **Buffer 300k의 역설적 하락:** 10M 스텝이라는 제한된 학습 예산 안에서 큰 버퍼는 **학습 초반에 버퍼가 오래된(stale) 경험으로 채워져 있는 기간이 길어지는 문제**를 유발한다. 버퍼가 클수록 현재 정책과 동떨어진 과거 정책의 경험이 더 오랫동안 잔류하게 되므로, Off-policy bias가 증가한다. 특히 본 실험처럼 `learning_starts=50,000`으로 초반에 버퍼를 무작위 정책으로 채울 때, 큰 버퍼는 이 무작위 데이터가 더 오래 영향을 미치게 한다.

#### 6.1.4 Frame Stacking

| 설정 | `frame_stack` | 최종 평균 점수 |
|---|---|---|
| 베이스라인 | 4 | 75.73 |
| `dqn_fs_1` | **1** | 4.66~9.11 |
| `dqn_fs_6` | **6** | 45.60 |

**결과:** Frame stacking을 제거하면 에이전트가 사실상 작동하지 않고(≈5점), 4→6으로 늘리면 소폭 하락한다.

**이론적 해석:** Breakout에서 공은 단일 프레임에서 위치만 알 수 있고 속도(방향)는 알 수 없다. 이는 Partial Observability 문제다. Frame stacking은 여러 프레임을 쌓아 속도와 방향 정보를 암묵적으로 인코딩한다. fs=1에서 거의 작동하지 않는 것은 이 Markov property 위반이 얼마나 치명적인지를 직접 증명한다. fs=6이 fs=4보다 낮은 이유는 중복 정보가 늘어나면서 NatureCNN의 학습이 오히려 복잡해지기 때문으로 보인다.

#### 6.1.5 미완료 실험 (설계 의도)

아래 실험들은 시간 제약으로 완료되지 못하였으나, 설계 의도를 명시한다.

- `dqn_lr_2.5e-4`: Nature DQN 원 논문의 RMSProp LR 값을 Adam에 적용. 1.5e-4가 최적이라면, 2.5e-4는 너무 공격적으로 불안정해질 것으로 예상된다.
- `dqn_expl_frac_0.2`: ε 감소 기간을 20%로 늘려 더 긴 탐색 페이즈 효과 측정.
- `dqn_target_10000`: Mnih 2015 원 논문값으로 복원. 5,000 대비 더 안정적일 것으로 예상.

---

### 6.2 PPO Ablation

#### 6.2.1 Clip Range

| 설정 | `clip_range` | 최종 평균 점수 |
|---|---|---|
| `ppo_clip_0.05` | 0.05 | 270.55 |
| **베이스라인** | **0.1** | **261.77** |
| `ppo_clip_0.2` | 0.2 | 263.76 |
| `ppo_clip_0.3` | 0.3 | **361.71** |

**결과:** clip_range를 크게 늘릴수록 성능이 향상되는 추세가 명확하다. 특히 0.3에서 361.71로 베이스라인 대비 38% 향상이 나타났다.

**이론적 해석:** Clip_range $\epsilon$은 정책 업데이트의 최대 허용 크기를 제어한다. Atari 게임처럼 희소한 보상(sparse reward)이 있는 환경에서는 좋은 경험이 드물게 발생하므로, 그 경험으로부터 최대한 많이 배우는 것이 유리하다. 작은 clip(0.05)은 update를 너무 조금씩 허용하여 학습이 느려지고, 큰 clip(0.3)은 좋은 신호를 적극적으로 활용하여 빠른 학습을 가능하게 한다. 다만, clip_range를 무한정 늘리면 기존 PPO의 trust region 보장이 사라지고 분산이 증가할 수 있어, 이 데이터만으로 "클수록 좋다"고 결론 짓는 것은 위험하다.

#### 6.2.2 GAE Lambda

| 설정 | `gae_lambda` | 최종 평균 점수 |
|---|---|---|
| `ppo_gae_0.9` | 0.90 | 331.21 |
| **베이스라인** | **0.95** | **261.77** |
| `ppo_gae_1.0` | 1.0 | **48.99** |

**결과:** λ=1.0(Monte-Carlo advantage)이 급격히 낮은 성능을 보였다.

**이론적 해석:** GAE(Generalized Advantage Estimation)에서 λ는 bias-variance tradeoff를 제어한다.
- λ=0: TD(0) advantage (높은 bias, 낮은 분산)
- λ=1: Monte-Carlo advantage (낮은 bias, 높은 분산)

Breakout은 에피소드 길이가 수천 스텝에 달할 수 있으며, 공이 벽을 뚫고 올라가 연속 점수를 얻는 장기 전략이 중요하다. 이런 환경에서 λ=1은 Monte-Carlo return의 **높은 분산**이 그래디언트 추정치를 불안정하게 만들어 학습이 발산에 가까워진다. λ=0.9(baseline 0.95 대비 더 많은 bias)가 오히려 안정적으로 331.21을 달성한 것은 분산 감소의 이점이 이 환경에서 크다는 것을 보여준다.

#### 6.2.3 N_epochs

| 설정 | `n_epochs` | 최종 평균 점수 |
|---|---|---|
| **베이스라인** | **4** | **261.77** |
| `ppo_ne_10` | 10 | **61.65** |

**결과:** n_epochs를 4→10으로 늘렸을 때 성능이 약 76% 하락하였다.

**이론적 해석:** PPO는 On-policy 알고리즘이다. 즉, 수집된 경험은 현재 정책(π_old)으로 만들어진 것이다. 같은 데이터를 여러 번 재사용(10 epochs)하면, 후반 업데이트에서는 **현재 정책 π와 데이터 수집 정책 π_old가 너무 달라져** importance ratio $r_t(\theta)$가 Clip 범위를 벗어난다. 이렇게 되면 업데이트가 신뢰할 수 없는 추정치를 기반으로 이루어지는 **policy drift** 현상이 발생하여 학습이 불안정해진다.

#### 6.2.4 Entropy 계수

| 설정 | `ent_coef` | 최종 평균 점수 |
|---|---|---|
| `ppo_ent_0.0` | 0.0 | 272.94 |
| **베이스라인** | **0.01** | **261.77** |
| `ppo_ent_0.05` | 0.05 | 346.14 |

**결과:** 엔트로피 보너스를 높이면 성능이 향상되고, 제거하면 비슷하거나 약간 향상된다.

**이론적 해석:** 엔트로피 보너스는 정책이 확률 분포를 균일하게 유지하도록 장려하여 **과조기 수렴(premature convergence)** 을 방지한다. ent_coef=0.0에서도 성능이 크게 떨어지지 않는다는 것은 baseline의 0.01이 이미 충분하다는 의미일 수 있다. ent_coef=0.05에서의 향상은 Breakout처럼 다양한 전략이 존재하는 환경에서 더 많은 탐색이 유리함을 시사한다.

#### 6.2.5 기타 PPO Ablation 결과 요약

| 설정 | 변경 내용 | 최종 평균 점수 | 해석 |
|---|---|---|---|
| `ppo_gamma_0.995` | γ 0.99 → 0.995 | 231.55 | 더 긴 시야가 오히려 학습 불안정 |
| `ppo_lr_1e-4` | lr 2.5e-4 → 1e-4 | 245.54 | 학습 속도 저하 |

---

### 6.3 A2C Ablation

#### 6.3.1 Advantage 정규화 (Normalize Advantage)

| 설정 | `normalize_advantage` | 최종 평균 점수 |
|---|---|---|
| **베이스라인** | **False** | **386.03** |
| `a2c_normadv` | **True** | **10.54** |

**결과:** Advantage 정규화를 활성화했을 때 성능이 386.03 → 10.54로 **97% 폭락**하였다. 본 실험의 가장 충격적인 결과 중 하나다.

**이론적 해석:** Advantage 정규화는 미니배치 내의 advantage를 평균 0, 분산 1로 표준화하는 기법으로, PPO에서는 안정적인 학습에 기여한다고 알려져 있다. 그러나 A2C에서는 왜 치명적인가?

A2C의 `n_steps=5`는 8개의 환경 × 5 스텝 = 40개의 전이만으로 advantage를 계산한다. **40개 샘플에서 추정한 평균과 표준편차는 통계적으로 매우 불안정**하다. 특히 에피소드 시작 직후처럼 advantage 값이 대부분 비슷한 상황에서는 표준편차가 0에 가까워져 수치 폭발(NaN/Inf)이 발생할 수 있다.

반면 PPO는 `n_steps=128 × 8 env = 1,024 개`의 advantage를 모아서 정규화하므로 통계적으로 훨씬 안정적이다. 즉, **Advantage 정규화는 충분한 배치 크기가 확보되어야 효과적이며, A2C의 짧은 rollout에서는 오히려 그래디언트 방향을 왜곡하는 부작용을 낳는다.**

#### 6.3.2 Value Function 계수 (vf_coef)

| 설정 | `vf_coef` | 최종 평균 점수 |
|---|---|---|
| **베이스라인** | **0.25** | **386.03** |
| `a2c_vf_0.5` | **0.5** | 356.50 |

**결과:** vf_coef를 PPO 기본값(0.5)으로 높여도 성능이 소폭 하락할 뿐, 크게 유지된다.

**이론적 해석:** vf_coef는 Actor-Critic 손실 함수에서 Critic(가치 함수) 업데이트의 가중치를 결정한다. A2C의 기본값 0.25는 가치 학습보다 정책 학습에 더 집중하는 설정이다. 0.5로 높이면 가치 함수의 예측 정확도는 올라가지만, 정책 업데이트 신호가 상대적으로 약해질 수 있다. 결과적으로 소폭 하락했지만 큰 차이가 없다는 것은 A2C에서 vf_coef가 비교적 덜 민감한 하이퍼파라미터임을 시사한다.

#### 6.3.3 옵티마이저 (Optimizer)

| 설정 | 옵티마이저 | 최종 평균 점수 |
|---|---|---|
| **베이스라인** | **RMSProp** | **386.03** |
| `a2c_adam` | **Adam** | 305.84 |

**결과:** Adam으로 교체하면 성능이 약 21% 하락한다.

**이론적 해석:** RMSProp은 A3C 원 논문에서 선택된 이유가 있다. 비동기 또는 단기 rollout 환경에서 그래디언트의 스케일이 빠르게 변하는데, RMSProp은 최근 그래디언트 제곱의 이동 평균으로 학습률을 조절하여 이런 비정상(non-stationary) 환경에 적합하다. Adam은 RMSProp에 Momentum을 추가한 것으로, 7e-4라는 A2C에 맞게 튜닝된 LR이 Adam에서는 다른 동작을 할 수 있다. 성능 하락은 LR 재튜닝 없이 옵티마이저를 교체했기 때문일 가능성이 높다.

#### 6.3.4 기타 A2C Ablation 결과 요약

나머지 A2C ablation은 시간 제약으로 별도 분석을 생략하였으나 주요 결과를 정리한다.

| 설정 | 변경 내용 | 최종 평균 점수 |
|---|---|---|
| `a2c_lr_2.5e-4` | lr 7e-4 → 2.5e-4 | (미완료) |
| `a2c_nstep_16` | n_steps 5 → 16 | (미완료) |
| `a2c_gae_0.95` | gae_lambda 1.0 → 0.95 | (미완료) |
| `a2c_nstep16_gae095` | n_steps=16 + gae_lambda=0.95 | (미완료) |

특히 `a2c_nstep16_gae095`는 A2C를 PPO와 유사한 어드밴티지 추정 방식으로 바꾸는 흥미로운 실험으로, 미완료가 아쉽다.

---

## 7. 종합 분석 및 이론적 반성

### 7.1 On-policy vs Off-policy: Breakout에서 A2C의 우위

10M 스텝 예산에서 On-policy 알고리즘(A2C, PPO)이 Off-policy 알고리즘(DQN)을 크게 앞서는 결과는 직관에 반할 수 있다. 일반적으로 Off-policy 알고리즘은 Replay Buffer를 통한 샘플 재사용으로 **샘플 효율이 높다**고 알려져 있기 때문이다.

그러나 이 결과를 이해하는 데 중요한 요소들이 있다:

1. **병렬 환경의 영향:** A2C와 PPO는 8개의 환경을 병렬로 실행하므로 실제로는 10M 스텝이지만 다양한 게임 상태를 8배 더 효율적으로 탐색한다. DQN은 1개의 환경만 사용하는 단일 스레드 구조다.

2. **업데이트 빈도:** A2C는 매 5×8=40 전이마다 업데이트한다. DQN은 4 스텝마다 1회 업데이트하지만, 처음 50,000 스텝은 업데이트 없이 버퍼를 채운다. 결과적으로 실제 그래디언트 업데이트 횟수 차이가 크다.

3. **Breakout의 보상 구조:** Breakout은 벽돌을 맞출 때마다 즉각적인 보상이 발생하는 **비교적 밀집된 보상(dense reward)** 구조다. 이런 환경에서는 On-policy 알고리즘도 학습 신호를 충분히 얻을 수 있어, Off-policy의 샘플 효율 우위가 줄어든다.

### 7.2 Double DQN이 DQN보다 낮은 이유

가장 의외의 결과는 Double DQN(33.10)이 표준 DQN(75.73)보다 낮다는 것이다. 이에 대한 가능한 설명:

1. **Exploration 관점:** 표준 DQN의 최대값 과대 추정(overestimation bias)은 실제로 에이전트가 더 적극적으로 탐색하게 만드는 부작용을 가질 수 있다. Q값이 과대 추정되면 더 다양한 행동에 대한 기대값이 높아져, 결과적으로 초기 탐색에 유리하게 작용할 수 있다.

2. **단일 시드 한계:** 본 실험은 seed=7의 단일 실행 결과다. Double DQN의 성능은 시드에 따른 분산이 클 수 있으며, 다수 시드 평균을 보면 이론적 예측(DDQN ≥ DQN)이 회복될 가능성이 있다.

3. **하이퍼파라미터 의존성:** Double DQN의 이점은 target_update_interval이 충분히 클 때 더 두드러질 수 있다. 본 실험의 target_update_interval=1,000은 지나치게 잦아서 Double DQN의 안정화 효과가 표준 DQN의 탐색적 bias를 충분히 상쇄하지 못했을 가능성이 있다.

### 7.3 하이퍼파라미터 민감도 비교

실험 전반을 통해 확인된 알고리즘별 하이퍼파라미터 민감도 수준:

| 알고리즘 | 민감도 | 주요 민감 파라미터 |
|---|---|---|
| DQN | **매우 높음** | lr, target_update_interval, buffer_size |
| PPO | 중간 | clip_range, gae_lambda, n_epochs |
| A2C | 중간 (특이점 존재) | normalize_advantage (파국적 실패), optimizer |

DQN은 하이퍼파라미터 변경에 극히 민감하여, lr 변화 하나로 2.5배 이상의 성능 차이를 보였다. 이는 Off-policy 학습 특성상 Replay Buffer의 데이터 분포와 타깃 네트워크의 안정성이 복잡하게 상호작용하기 때문이다.

PPO는 n_epochs=10처럼 명백히 잘못된 설정을 제외하면 상대적으로 견고(robust)하다. Clip 제약이 과도한 업데이트를 차단하므로, 하이퍼파라미터 변경이 파국적으로 작용하기 어렵다.

A2C는 normalize_advantage처럼 이론적으로 무해해 보이는 변경이 파국적 결과를 낳을 수 있음을 보여주었다. 이는 짧은 rollout(n_steps=5)으로 인한 통계적 불안정성이 숨겨진 위험 요소임을 시사한다.

### 7.4 한계와 신뢰성

본 실험의 한계점:

1. **단일 시드:** 모든 실험이 seed=7 단일 실행이다. RL 학습은 시드에 따른 분산이 매우 크므로, 일부 결과(특히 DDQN의 낮은 성능)는 재현성 문제일 수 있다.
2. **10M 스텝 제약:** DQN 계열은 일반적으로 더 긴 학습(50M+ 스텝)에서 On-policy 알고리즘을 따라잡는다. 본 결과는 짧은 예산에서의 특수 케이스다.
3. **튜닝된 베이스라인의 부재:** PPO와 A2C가 DQN보다 좋다는 결론이 하이퍼파라미터 선택의 영향을 받을 수 있다. DQN의 best 하이퍼파라미터(lr=1.5e-4, target=5000 조합)라면 순위가 달라질 수 있다.

---

## 8. 결론

### 8.1 주요 발견 요약

1. **알고리즘 순위 (10M 스텝 기준):** A2C(386) > PPO(262) >> DQN(76) > DDQN(33)
2. **가장 극적인 Ablation 결과:** DQN lr 1.5e-4에서 75.73 → 191.17 (2.5× 향상), A2C normalize_advantage에서 386 → 10 (97% 하락)
3. **이론과 실험의 불일치:** Double DQN의 이론적 우위가 단일 시드 10M 스텝에서는 관찰되지 않았다.
4. **A2C normalize_advantage 실패:** 배치 크기(40개)가 너무 작을 때 advantage 정규화는 도움이 아닌 해가 된다.
5. **DQN의 하이퍼파라미터 민감도:** Target network 업데이트 주기와 learning rate가 DQN 성능에 가장 큰 영향을 미친다.

### 8.2 실무적 시사점

- **제한된 예산 환경:** On-policy 알고리즘(특히 A2C, PPO)이 Off-policy보다 구현과 튜닝이 쉽고 안정적이다.
- **DQN 사용 시:** target_update_interval을 5,000~10,000으로 유지하고, learning rate를 신중하게 선택하는 것이 중요하다.
- **PPO 튜닝:** clip_range를 늘리는 것이 Atari 환경에서 효과적일 수 있으며, n_epochs를 4 이상으로 높이는 것은 위험하다.
- **A2C 주의사항:** normalize_advantage는 반드시 해제해야 하며, optimizer 변경 시 learning rate 재튜닝이 필요하다.

### 8.3 향후 방향

- **다수 시드 반복 실험:** 신뢰 구간 기반의 통계적으로 유의한 비교가 필요하다.
- **DQN 최적 설정 조합:** lr=1.5e-4 + target_update=5,000 조합의 성능 확인.
- **PPO clip_range 탐색:** clip_range=0.3 이상의 영역에서 성능 포화점 확인.
- **Rainbow DQN:** Prioritized Experience Replay, n-step return 등을 추가하여 DQN 계열의 한계 극복 시도.

---

*본 실험에 사용된 코드는 [GitHub Repository](.)에서 확인할 수 있습니다.*

# Linear probes: 무엇이 밝혀졌고, 무엇을 아직 말할 수 없는가

문헌 확인일: **2026-09-20**. ARENA 1.3.1의 세 주제인 truth, deception, high-stakes probing에 직접 영향을 주는 9개 1차 문헌만 추렸다. 체계적 문헌고찰이나 학계 전체의 합의 조사로 읽으면 안 된다. 아래의 **확인됨**은 특정 결과·반례가 보고되었다는 뜻이며, 모든 모델에서 성립한다는 뜻이 아니다. 2026년 논문의 새 결과는 **조건부 증거**로 취급한다.

이 장에서 가장 중요한 구분은 다음과 같다.

> 라벨을 읽어낼 수 있음 → 새로운 분포에서도 읽어냄 → 개입하면 행동이 변함 → 원래 계산에서 그 의미로 사용함 → 적대적 상황에서도 믿을 만함.

왼쪽 결과 하나만으로 오른쪽을 결론 내릴 수 없다. 이 구분은 아래 문헌들을 종합한 **수업의 해석 원칙**이다.

## 주장별 판정과 실험 설계

| 자주 하는 주장 | 현재 근거와 판정 | 이번 실습에서 바꿀 것 |
|---|---|---|
| “선형 probe가 성공했으니 진실 표상을 찾았다.” | **조건부.** 간단한 사실 문장의 truth 라벨을 선형적으로 읽고, 일부 데이터셋 간 전이와 개입 효과를 얻은 결과가 있다. 이것은 모든 의미의 ‘진실’에 통하는 증명이 아니다. [Marks & Tegmark, §2–6](https://arxiv.org/html/2310.06824v3) | 먼저 “어떤 데이터·모델·층·토큰에서 어떤 라벨을 예측했다”고 쓴다. random-label, 문자열 baseline, 도메인 전이 평가를 붙이는 것은 이 수업의 설계 권고다. |
| “PCA에서 두 무리가 보이면 truth를 비지도 발견했다.” | **과도한 해석.** PCA는 분산을 찾는다. 그림에 정답으로 색을 칠하거나 보기 좋은 축을 골랐다면 해석 단계에 감독 정보가 들어간다. 또한 CCS의 일관성 목적함수도 truth만을 식별하지 못한다는 이론적 반례와 실험이 있다. PCA와 CCS는 다른 방법이다. [Farquhar et al.](https://arxiv.org/abs/2312.10029v2) | PCA는 탐색용으로 한정한다. probe 학습·층 선택·최종 평가는 분리한다. “비지도 truth discovery”를 입증하는 필수 실습으로 CCS를 넣지 않는다. |
| “layer 14가 truth layer다.” | **일반화 불가.** 원 논문의 특정 설정이다. 2026년 후속 연구에서는 factual/reasoning 과제, 난도, 층, instruction template에 따라 방향과 전이성이 달랐다. 보편성의 범위를 좁히는 증거이지, truth probing 전체를 반박한 결과는 아니다. [Poulis et al., v2](https://arxiv.org/abs/2604.03754v2) | validation에서 layer sweep을 한 뒤 test를 한 번 본다. 모델을 바꾸면 층도 재선정한다. template 변경을 별도 OOD 조건으로 둔다. |
| “그 방향으로 steering되면 모델의 실제 reasoning을 설명했다.” | **충분조건 아님.** subspace patching으로 원하는 출력 변화를 내면서도 원래의 계산 경로를 잘못 귀속시키는 반례가 있다. [Makelov et al., ICLR 2024](https://openreview.net/pdf?id=Ebt7JgMHv1) 다만 이 ‘illusion’의 정의와 평가 방식에는 반론도 있다. [Wu et al.](https://arxiv.org/abs/2401.12631v1) | 먼저 “이 위치·크기의 개입에 출력이 민감하다”고 결론낸다. random direction, 같은 norm, 양·음 방향과 세기, 다른 행동 손상 등을 비교한다. 이 통제들을 통과해도 원래 계산의 완전한 설명으로 승격하지 않는다. |
| “간단한 instructed pairs에서 배운 probe로 전략적 기만을 잡을 수 있다.” | **가능성을 보인 조건부 결과.** Apollo는 Llama-3.3-70B-Instruct와 정해진 평가 시나리오에서 강한 전이를 보고했다. 저자들도 spurious correlation, 애매한 라벨, 비교적 노골적인 기만이라는 한계를 적었다. [Goldowsky-Dill et al., §5](https://arxiv.org/html/2502.03407v1) | 8B 실습의 성공을 70B 논문의 수치 재현으로 부르지 않는다. 훈련에 없던 scenario·instruction 표현을 평가하고, 같은 사실의 paired examples는 같은 split에 묶는다. |
| “truth probe는 deception detector다.” | **이 등치는 성립하지 않음.** 참인 말로도 상대를 오도할 수 있다. 2026년 연구는 3개 공개 LLM에서 거짓말 없는 기만을 실험하고, 표준 true/false 데이터로 학습한 truth probe의 탐지 공백을 보고했다. [Thormann](https://arxiv.org/abs/2603.10003v1) | factual falsehood, 역할극/지시, 실제 기만 행동을 서로 다른 라벨로 다룬다. 참이지만 오도하는 예시를 별도 holdout으로 둔다. 이 논문이 모든 deception probe의 실패를 입증한 것은 아니다. |
| “probe를 통과하도록 학습시키면 안전해진다.” | **미해결이며 반례가 있음.** RL-Obfuscation에서는 행동을 유지하면서 token-level monitor를 피하는 학습이 가능했다. 같은 연구의 max/attention 기반 monitor는 더 강했지만, 그것은 평가한 공격·모델 범위의 결과다. [Gupta & Jenner, v4](https://arxiv.org/abs/2506.14261v4) | 정적인 OOD 성능과 회피하도록 최적화된 모델에 대한 성능을 분리해 쓴다. 이번 입문 실습에서는 공격용 RL 학습을 생략하고 “적대적 강건성 미평가”를 결과의 한계에 적는다. |
| “attention probe가 가장 좋고, 높은 AUROC면 실용적이다.” | **조건부.** high-stakes 논문은 dev에서 Attention을 골랐지만 test에서는 Softmax가 조금 높고, 소량 데이터에서는 Mean이 강했다. 최고 probe도 1% FPR에서 TPR 약 43%였다. [McKenzie et al., §3](https://arxiv.org/html/2506.10805v4) | last/mean/attention을 동일 split·선택 예산으로 비교한다. AUROC와 함께 낮은 FPR에서의 TPR을 본다. 작은 음성 표본으로 1% FPR을 안정적으로 추정했다고 주장하지 않는다. |

## Attention 실습에서 꼭 남길 두 가지

**High-stakes는 기만 의도가 아니다.** 논문의 대상은 상호작용이 큰 피해로 이어질 가능성을 시사하는 텍스트이며, GPT-4o 라벨과 사람 검증에도 개념·라벨의 주관성이 남는다. 정상적인 긴급 도움 요청도 high-stakes일 수 있다. 논문의 큰 계산 절약은 이미 실행 중인 모델의 activation을 재사용하는 조건이다. 로컬 실습 전체의 비용을 probe head 비용과 혼동하면 안 된다. [McKenzie et al., §2, §6](https://arxiv.org/html/2506.10805v4)

**Attention weight만으로 분류 근거를 읽지 않는다.** 단일 head의 점수는 `Σᵢ attentionᵢ × value_scoreᵢ + bias`이다. 큰 weight가 음의 value와 결합할 수도 있다. 원 논문의 Figure 7도 query와 value를 함께 분석한다. 이 식에서 직접 따라오는 수업 권고는 weight, value, 두 값의 곱을 나란히 보는 것이다. 그래도 이는 probe 계산의 설명이며, 원 LLM의 행동 원인을 입증하지 않는다. [McKenzie et al., Fig. 7, Appendix A.1](https://arxiv.org/html/2506.10805v4)

## 저장소 오류와 연구 결과의 수정은 구별한다

이 fork의 `chapter1_transformer_interp/exercises/part31_linear_probes/solutions.py`에는 **2026-08-29 정정** 주석들이 이미 들어 있다. 그중 `2603.10003`을 **Berger (2026)**로 적은 부분은 **Tom-Felix Thormann**으로 고쳐 읽어야 한다. 저자명 오류는 서지 오류이며 논문 결과가 반박되었다는 뜻이 아니다. [실제 논문 메타데이터](https://arxiv.org/abs/2603.10003v1)

반면 “layer 14의 보편성”, “거짓말 탐지가 곧 모든 기만 탐지”, “steering 성공만으로 내부 의미 확정” 같은 확대 해석을 제한하는 것은 **결론의 범위에 대한 수정**이다. 원 논문의 특정 실험 결과가 모두 debunk되었다는 식으로 가르치지 않는다. 위 표의 조건을 만족한 결과는 그 범위 안에서 여전히 유용하다.

## 지금 읽을 범위와 미룰 범위

필수는 표의 주장과 그 옆의 실험 변경이다. 직접 읽을 논문은 Geometry of Truth의 결과 그림, Apollo의 §5 한계, high-stakes의 §3·§6을 우선한다. 2026년 후속 논문은 해당 실습에서 결론을 쓸 때 초록과 관련 결과를 확인하면 된다.

CCS 목적함수의 전체 증명, subspace illusion 논쟁의 정의론, second-order belief probe 구현, adversarial RL 재현은 지금은 미룬다. 각각 중요한 문제지만 이번 장의 목표인 **좋은 대조 실험을 설계하고 주장 범위를 정하는 능력**을 익히는 데 선행 구현이 필요하지 않다.

## 확인한 1차 문헌과 버전

날짜는 확인한 arXiv 버전 기준이다. 학회판은 별도로 표시했다. **버전 날짜가 학회 발표 날짜와 같은 것은 아니다.** 모든 링크는 2026-09-20에 확인했다.

| 문헌 | 확인한 버전·날짜 | 이 수업에서의 역할 |
|---|---|---|
| Samuel Marks & Max Tegmark, *The Geometry of Truth* | [arXiv v3, 2024-08-19](https://arxiv.org/abs/2310.06824v3); COLM 2024 | truth probing의 원래 긍정적 근거 |
| Nicholas Goldowsky-Dill et al., *Detecting Strategic Deception Using Linear Probes* | [arXiv v1, 2025-02-05](https://arxiv.org/abs/2502.03407v1) | instructed pairs와 기만 시나리오 전이; 저자들의 한계 설명 |
| Alex McKenzie et al., *Detecting High-Stakes Interactions with Activation Probes* | [arXiv v4, 2026-01-23](https://arxiv.org/abs/2506.10805v4); NeurIPS 2025 | pooling 비교, 낮은 FPR 성능, label·비용 조건 |
| Sebastian Farquhar et al., *Challenges with unsupervised LLM knowledge discovery* | [arXiv v2, 2023-12-18](https://arxiv.org/abs/2312.10029v2) | CCS의 목적함수가 truth를 유일하게 식별하지 않는 반례 |
| Aleksandar Makelov et al., *Is This the Subspace You Are Looking for?* | [ICLR 2024 학회판](https://openreview.net/pdf?id=Ebt7JgMHv1); [arXiv v2, 2023-12-06](https://arxiv.org/abs/2311.17030v2) | 개입 성공과 원래 계산에 대한 설명 사이의 간극 |
| Zhengxuan Wu et al., *A Reply to Makelov et al. (2023)'s “Interpretability Illusion” Arguments* | [arXiv v1, 2024-01-23](https://arxiv.org/abs/2401.12631v1) | illusion의 정의·평가를 둘러싼 반론; 논쟁을 일방적 합의로 표현하지 않기 |
| Angelos Poulis, Mark Crovella & Evimaria Terzi, *Testing the Limits of Truth Directions in LLMs* | [arXiv v2, 2026-09-11](https://arxiv.org/abs/2604.03754v2); 메타데이터에 BlackboxNLP 2026 표기 | 층·과제·template에 따른 전이성의 한계 |
| Tom-Felix Thormann, *Probing the Limits of the Lie Detector Approach to LLM Deception* | [arXiv v1](https://arxiv.org/abs/2603.10003v1); 페이지의 submission history는 2026-02-16 표기 | 거짓말 없는 기만과 truth probe의 탐지 공백 |
| Rohan Gupta & Erik Jenner, *RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?* | [arXiv v4, 2026-02-26](https://arxiv.org/abs/2506.14261v4) | monitor 회피 학습과 집계 방식별 차이 |

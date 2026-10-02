# [1.3.1] Linear Probes — 한국어 재구성판

**probe의 성공을 어떻게 증거로 만드는가, 그리고 그 증거를 어떻게 다시 허무는가.**

원본 ARENA [1.3.1] Linear Probes(`../part31_linear_probes/`)를 다시 쓴 장이다. 원본은 *Geometry of Truth*를 재현하는 순서로 진행한다. 이 판은 같은 재료를 두 번 지나간다.

1. **1부 — 주장 세우기 (①~⑤):** 논문을 쓰는 사람의 순서. 관찰 → 대안 설명 배제 → 일반화 → 인과 개입 → 초록.
2. **2부 — 주장 허물기 (⑥~⑧):** 심사위원의 순서. 부정문, "읽힘 ≠ 쓰임", 감시 도구로서의 실패 모드, 2024–2026 연구로 1부의 주장을 다시 시험하고 살아남은 범위만 남긴다.

모든 notebook은 **Llama-3.1-8B-Instruct 하나**로 실제 실행했고, 본문의 숫자는 저장된 출력과 맞춰 썼다.

## 순서와 질문

| notebook | 부 | 묻는 것 | 저장된 출력의 핵심 | 실행 시간* |
|---|---|---|---|---|
| [① probe는 무엇을 묻는가](01_probe가_묻는_질문.ipynb) | 1부 | probe가 직접 답하는 질문, 왜 선형인가, MM vs LR, 층 sweep | layer 12 test AUROC 1.000, PC1이 분산의 56% | 1분 |
| [② 성공을 증거로 만들기](02_성공을_증거로_만들기.ipynb) | 1부 | 표면 통계·확률·probe 용량이라는 대안 설명 | 표면 baseline ≤0.48, random-init ≤0.58, **MM 귀무분포가 0과 1 양끝에 몰리는 이유** | 2분 |
| [③ 일반화](03_일반화.ipynb) | 1부 | cities 방향이 다른 주제·형식·언어·역할극 거짓말로 옮겨 가는가 | AUROC 0.81–1.00, 그러나 threshold는 전이되지 않음 | <1분 |
| [④ 인과 개입](04_인과_개입.ipynb) | 1부 | 그 방향으로 밀면 판단이 바뀌는가, 무작위 방향과 다른가 | α=0.5: 84%/99% 뒤집힘 vs 무작위 1%/0.2%; GoT 설정(α=1)은 비특이적 | 10분 |
| [⑤ 논문 초안](05_논문_초안.ipynb) | 1부→2부 | 증거 사다리, 초록, 한계, 심사위원이 찌를 곳 | (새 실험 없음) | <1분 |
| [⑥ 반박 1 — 부정문과 극성](06_반박1_부정문과_극성.ipynb) | 2부 | "보편적 진실 방향"은 부정문에서도 살아남는가 | 부정문에서 순위가 뒤집힘(0.005); t_G/t_P 분해로 회복(≥0.93) | <1분 |
| [⑦ 반박 2 — 읽힘과 쓰임](07_반박2_읽힘과_쓰임.ipynb) | 2부 | 만들어 낸 개념, LEACE 제거, 착시 논쟁, 비선형 표현 | 무작위 국가 묶음도 0.99; 올바른 분포의 제거만 판단을 무너뜨림 | 2분 |
| [⑧ 반박 3 — 감시 도구와 최종 판정](08_반박3_모니터와_최종판정.ipynb) | 2부 | 운영점, 맥락 이동, 회피 공격, 2026년 현장, 최종 장부 | AUROC 0.99인데 정직 응답 49% 오경보; soft prefix로 threshold 회피 | 4분 |

\* RTX A6000 기준, activation cache가 있을 때. 처음 실행하면 ①에서 cities 32개 층, ③·⑥에서 다른 데이터셋의 activation을 계산하느라 몇 분이 더 걸린다.

각 notebook은 같은 형식을 따른다.

- **학습 목표, 필수 경로, stop rule** — 다음 notebook으로 넘어가도 되는 기준
- **Cell NN · 제목** — 셀마다 `이번 셀의 질문 / 출력에서 볼 것 / 해석 범위 / 다음 단계의 목적`
- **✏️ Exercise** — scratch 셀에서 먼저 구현하고 `MY_IMPL = True`로 바꾸면 `tests.py`가 reference와 비교한다. 바로 아래에 실행 가능한 reference가 있다.
- **주장 장부** — notebook 끝마다 주장의 상태(`관찰됨 → 지지됨 → 유지 / 범위 축소 / 기각`)를 갱신한다. ⑤와 ⑧이 `results/*.json`을 모아 읽는다.

## 실행 환경

이 장의 출력은 다음 환경에서 만들었다: Vast.ai RTX A6000 48 GiB, Python 3.12.14, PyTorch 2.11.0+cu128, transformers 5.18.0 (`requirements.txt`).

```bash
# 1) GPU에 맞는 torch를 먼저 설치한 뒤 나머지
uv pip install -r chapter1_transformer_interp/exercises/part31_linear_probes_ko/requirements.txt

# 2) gated model 접근: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 에서 승인을 받은 계정으로
hf auth login          # 토큰을 notebook이나 저장소 파일에 쓰지 않는다

# 3) notebook을 ① → ⑧ 순서로 실행 (⑤·⑧은 앞 notebook의 results/*.json을 읽는다)
```

- **GPU 메모리:** bf16 8B 모델 약 15 GiB + activation. 24 GiB 이상 권장. 모델은 한 번에 하나만 올린다(②의 random-init 모델은 학습된 모델을 내린 뒤 올린다).
- **activation cache:** `cache/`(git 제외)에 float16으로 저장된다. 전체 약 3–4 GB. 지워도 다시 계산된다. 같은 이름에 다른 문장 목록이 들어오면 내용 hash가 달라져 자동으로 다시 계산한다.
- **CPU 주의:** 컨테이너의 CPU 할당량보다 BLAS thread가 많으면 4096차원 SVD가 수 분씩 걸린다. `probe_lab.py`는 import할 때 BLAS thread를 4개로 묶고, PCA·LEACE·LR은 torch(GPU)로 계산한다.
- **다른 모델:** `probe_lab.DEFAULT_MODEL`만 바꾸면 Llama·Qwen 계열에서 코드는 돈다. 숫자와 층 번호는 달라지므로 ①의 층 선택부터 다시 한다(다른 모델은 검증하지 않았다).

## 파일

| 파일 | 내용 |
|---|---|
| `probe_lab.py` | reference 구현과 실험 도구: block hook activation 추출, MM/LR probe(raw 좌표), group split, 귀무분포, bootstrap, residual 개입·편집 hook, LEACE, t_G/t_P 분해, 그림 규약 |
| `tests.py` | exercise 확인용 테스트 |
| `data/` | Truth-is-Universal 데이터(MIT). 출처는 `data/README.md` |
| `results/` | notebook별 핵심 수치(JSON)와 ⑧의 on-policy 생성 응답 |
| `cache/` | activation cache (git 제외) |

## 원본·이전 판과 달라진 점

**원본 ARENA [1.3.1]과 비교**

| 원본 | 이 판 |
|---|---|
| Llama-2-13B(§1–3) + Llama-3.1-8B-Instruct(§4) | Llama-3.1-8B-Instruct 하나 |
| `probe_layer=14`를 논문 설정에서 가져옴 | validation 분리도(d′)로 test를 보기 전에 선택 → layer 12 |
| PCA·probe·개입을 재현 순서로 나열 | 대안 설명 표(A1–A7)를 목차로 삼아 "어떤 실험이 어떤 설명을 배제하는가"로 배열 |
| 개입에 무작위 방향 대조군 없음 | 같은 norm의 무작위 방향 20개, 용량–반응, 층 띠 비교 |
| 부정문·제거 실험·회피 공격 없음 | ⑥ 극성 분해, ⑦ LEACE amnesic probing, ⑧ soft-prefix 회피 |
| deception probe의 AUROC 위주 평가 | 운영점(FPR) 평가와 맥락 이동 |
| (§5 attention probe) | ⑧에서 문헌으로만 다룸(McKenzie et al., Gupta & Jenner, GDM 2026) |

**이전 판(`study/linear_probes_ko`, GPT 작성)과 비교**

- 위치를 `study/`에서 chapter 1 안(`exercises/part31_linear_probes_ko/`)으로 옮기고 내용을 새로 썼다. 이전 판은 git 기록(commit `f37d6a45`)에 남아 있다.
- 이전 판에는 "probe의 성공을 왜 증거로 받아들이는가"를 쌓는 단계가 약했다. 1부(①~⑤)가 그 부분이다.
- 이전 판의 shuffled-label 대조군(20회 중 3회가 AUROC 1.0)은 해석되지 않은 채 남아 있었다. ②에서 같은 현상을 재현하고, MM probe의 귀무분포가 데이터 공분산 N(0, Σ)에서 뽑은 무작위 방향의 분포와 같다는 것으로 설명한다.
- 원본 ARENA의 few-shot 예시(Tokyo, Hanoi, 'gato', 'aire')가 평가 데이터에 들어 있어, 데이터셋 어디에도 없는 예시(Bergen, Graz, 'queso', 'espejo')로 바꿨다.

## 주요 참고문헌

notebook마다 링크가 있다. 2026년 문헌은 arXiv 메타데이터와 초록으로 확인했다(2026-10-02).

- 출발점: Marks & Tegmark, *The Geometry of Truth* (COLM 2024) · Goldowsky-Dill et al., *Detecting Strategic Deception Using Linear Probes* (ICML 2025)
- 방법론 비판: Hewitt & Liang 2019 · Belinkov 2022 · Ravichander et al. 2021 · Elazar et al. 2021 · Belrose et al. 2023 (LEACE) · Makelov et al. 2024
- 진실 probe의 범위: Levinstein & Herrmann 2024 · Bürger et al. 2024 · Orgad et al. 2025 · Liu et al. 2024 · Ying et al. 2026 · Poulis et al. 2026 · von Klinski et al. 2026
- 비선형 표현: Engels et al. 2025 · Csordás et al. 2024 · Nanda, Lee & Wattenberg 2023
- 감시 도구: McKenzie et al. 2025 · Kirch et al. 2025 · Kretschmar et al. 2025 · Cunningham et al. 2026 (Anthropic) · Kramár et al. 2026 (GDM) · Cooney et al. 2026 · Fomin et al. 2026
- 회피 공격: Bailey et al. 2024 · Gupta & Jenner 2025 · McGuinness et al. 2025 · Keenan et al. 2026

## 한계

이 장의 판정은 모두 **한 모델(Llama-3.1-8B-Instruct), 한 probe 층, 단순 사실 문장, 하나의 few-shot 형식, 작은 운영 표본(정상 응답 50개)** 안에서의 판정이다. 여러 모델·크기, on-policy 기만 데이터, 믿음이 검증된 model organism, weight 수준의 적대적 학습은 다루지 않았다. ⑧의 끝에 이 범위를 넘으려면 필요한 실험을 정리했다.

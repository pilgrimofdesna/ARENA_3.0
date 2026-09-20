# Linear Probes — 한국어 실험 설계 수업

**[00_START_HERE.ipynb](00_START_HERE.ipynb)부터 여세요.** ARENA 1.2를 아는 학습자를 위해 `[1.3.1] Linear Probes`를 재구성했습니다. 기본 개념을 반복하기보다 예측·대조군·분포 변화·허용되는 결론을 연습합니다.

| 순서 | 학습 자료 | 목표 |
|---|---|---|
| 0 | [시작 안내](00_START_HERE.ipynb) | 목표와 실행 환경 확인 |
| 1 | [설계와 probe](01_design_and_probes.ipynb) | 누수 없는 split, MM/LR, PCA, shortcut 반례 |
| 2 | [실제 truth 실험](02_real_truth_experiment.ipynb) | Llama-2-13B의 IID/OOD 결과와 신뢰구간 해석 |
| 3 | [개입과 주장](03_intervention_and_claims.ipynb) | readout·steering·자연 계산의 인과 설명 구분 |
| 4 | [Deception과 attention](04_deception_attention.ipynb) | 지시문/기만/위험도 라벨 구분, pooling 대조 실험 |
| 참고 | [최신 근거와 수정할 주장](SOURCES_AND_CLAIMS.md) | 반복된 근거·조건부 결과·논쟁 중인 해석 구분 |

각 notebook의 질문에 먼저 답한 후 다음 셀을 실행하세요. “논문의 숫자를 맞히기”가 아니라 **틀릴 수 있는 주장을 쓰고, 이를 구분할 실험을 고르는 것**이 목표입니다. Chat에서 가르침을 이어갈 때는 `01의 실험 2까지 했고 내 해석은 …`처럼 알려 주세요.

## 이 인스턴스에서 실행

VS Code에서 `/root/ARENA_3.0`을 열고 notebook kernel을 **ARENA · Vast PyTorch** (`/venv/main/bin/python`)로 선택합니다. 기본 00~03은 작은 합성 데이터와 이미 계산한 activation cache를 사용합니다. GPU 모델 재로딩이나 HF 토큰 입력이 필요하지 않습니다. 04의 기본 경로도 합성 데이터이며 실제 8B 경로는 명시적으로 켭니다.

현재 환경은 RTX A6000 약 48GB / Python 3.12 / PyTorch 2.11.0+cu128입니다. CUDA tensor 연산을 확인했습니다. 드라이버나 시스템 CUDA는 변경하지 않았습니다. 이 챕터에 필요한 패키지만 `/venv/main`에 설치했습니다. repository 전체 `uv sync` 또는 `install.sh`는 실행하지 않았습니다. 그 설정은 이 실습에 필요 없는 의존성과 다른 PyTorch 버전을 요구합니다.

실제 truth 실험을 다시 계산하려면:

```bash
cd /root/ARENA_3.0/study/linear_probes_ko
/venv/main/bin/python -u run_truth_pilot.py
```

기존 `artifacts/`를 같은 seed 결과로 덮어씁니다. 새 실험이면 먼저 다른 폴더에 복사해 비교하세요. 새 프로세스가 종료되면 GPU 메모리는 반환됩니다. 기본 학습에는 재계산이 필요 없습니다.

재실행은 VS Code에 종속되지 않는 **supervisor 등록 작업**으로 관리하면 접속 종료에도 지속할 수 있습니다. 이번 작업에서는 외부 서비스/포트를 추가하지 않았습니다. 노트북의 일반 실행은 kernel과 연결 관리 방식에 따라 중단될 수 있으므로, “VS Code를 닫아도 모든 notebook 실행이 계속된다”고 가정하지 마세요.

## 실제 실험과 학습용 반례 구분

- `artifacts/activations.npz`: 실제 Llama-2-13B의 decoder block `[8,14,20,28,36]` 출력. 학습한 모델은 고정하고 probe만 학습했습니다.
- `artifacts/data_manifest.csv`: 468문장과 domain/group/split. cities 240, neg_cities 48, 번역 60, 수 비교 120. 작은 단일 seed pilot입니다.
- `artifacts/results.json`: 모델 snapshot, source/data commit, 실행 환경, 설정, 결과 및 한계.
- `artifacts/validation_selection.csv`: validation으로만 고른 layer/C.
- 01의 shortcut/XOR와 03 첫 반례, 04 기본 sequence task는 **합성 실험**입니다. LLM의 진실·기만 표현을 실증한 결과가 아닙니다.

원본 번역 CSV에는 같은 positive statement 3개의 중복이 있었습니다. 라벨 충돌이 없는지 검사한 뒤 statement를 중복 제거하고 단어 단위로 표본을 뽑았습니다. cities의 같은 도시와 negated counterpart를 서로 독립된 표본으로 취급하지 않습니다.

## 원본과 달라진 점

| 원본 범위 | 재구성 위치 | 변경 이유 |
|---|---|---|
| §1 activation/PCA/layer sweep | 01, 02 | train-only PCA; test로 층을 선택하지 않음; block hook으로 메모리 절약 |
| §2 MM/LR/일반화/CCS | 01, 02 | MM midpoint·LR raw 좌표 수정; CCS의 식별 한계는 개념만 |
| §3 causal interventions | 03 | additive steering을 NIE라 부르지 않음; 위치/scale/대조군 명시 |
| §4 deception | 04 | fact 단위 split; 지시문 label과 실제 행동 label 분리 |
| §5 high-stakes attention | 04 | 동일 split baseline, padding mask, 비선형 pooling, weight/value 구분 |

70B 모델, quantized detector pickle 로딩, 적대적 RL 훈련, CCS 전체 구현은 기본 과정에서 뺐습니다. 현재 학습 목표에 비해 계산 비용이나 주변 설명이 큽니다. 최신 연구의 적용 범위는 [출처 문서](SOURCES_AND_CLAIMS.md)에 정리했습니다. **후속 논문 하나를 학계 합의로, 원본 구현 오류를 분야 전체의 debunk로 취급하지 않습니다.**

## 재현·검증 파일

- `probe_lab.py`, `test_probe_lab.py`: 정확한 좌표·분할·hook·padding을 점검하는 작은 구현과 테스트.
- `build_notebooks.py`: 00~03의 원본 텍스트. 실행하면 notebook 출력이 초기화됩니다.
- `run_truth_pilot.py`: 실제 GPU 실험.
- `requirements-study.txt`, `environment_versions.json`: 필요한 패키지와 실제 설치 버전.
- `models.json`, `prepare_models.py`: 새 인스턴스에서 동일 snapshot을 명시적으로 다운로드하는 도구. 예: `/venv/main/bin/python prepare_models.py --model truth`. 토큰은 숨김 입력 또는 `HF_TOKEN` 환경변수로만 받고 저장하지 않습니다.
- `VALIDATION.md`: 완료한 검사와 실제 결과 요약. 실행하지 않은 범위도 표시합니다.

테스트:

```bash
/venv/main/bin/python -m unittest discover -s /root/ARENA_3.0/study/linear_probes_ko -p test_probe_lab.py -v
```

## 인증과 보관

HF 모델은 제공된 인증으로 다운로드했지만 토큰을 notebook, 저장소 파일, Hugging Face 로그인 저장소에 저장하지 않았습니다. `model_paths.local.json`에는 인증 정보 없이 로컬 snapshot 경로만 있으며 Git에서 제외했습니다. 모델은 safetensors 및 로컬 코드만 사용합니다. 이 수업은 OpenAI API를 호출하지 않습니다.

다른 인스턴스에서는 그 GPU에 맞는 PyTorch를 먼저 확인한 다음 `uv pip install --python /venv/main/bin/python -r requirements-study.txt`로 실습 패키지를 설치하고, 위 kernel 등록을 재현하려면 `python -m ipykernel install --user --name arena-vast --display-name 'ARENA · Vast PyTorch'`를 실행합니다. 현재 인스턴스에서는 이미 끝났습니다.

Vast의 이 인스턴스는 영구 volume이 없습니다. **stop/start는 보존되지만 recycle/destroy 시 디스크가 지워집니다.** 제공한 ZIP을 로컬로 내려받아 보관하세요. ZIP에는 모델 가중치를 포함하지 않습니다. 데이터/activation 분석은 ZIP으로 복구할 수 있고, 새 GPU 추론은 HF 권한과 모델 재다운로드가 필요합니다.

## 출처

- 원본 fork: [pilgrimofdesna/ARENA_3.0](https://github.com/pilgrimofdesna/ARENA_3.0), 사용 commit `4605b1fb676dcf2ba0704c7821b19a0a58f4484a`.
- 원본 장: `chapter1_transformer_interp/instructions/pages/11_[1.3.1]_Linear_Probes.md` 및 `exercises/part31_linear_probes/solutions.py`.
- 데이터: [Geometry of Truth](https://github.com/saprmarks/geometry-of-truth), [Apollo deception-detection](https://github.com/ApolloResearch/deception-detection).
- 논문·후속 반례·정확한 버전은 [SOURCES_AND_CLAIMS.md](SOURCES_AND_CLAIMS.md) 참조.

원본 파일은 바꾸지 않았습니다. 이 폴더의 설명·실험 코드는 개인 학습용 재구성이며 저자들이 검토한 공식 ARENA 교재는 아닙니다.

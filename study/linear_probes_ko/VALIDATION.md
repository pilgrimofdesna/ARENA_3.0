# 검증 기록

확인일: 2026-09-20. 아래는 실제 실행 결과이며 논문 전체 재현이나 통계적으로 확정된 일반 결론이 아닙니다.

## 환경과 구현

- RTX A6000에서 PyTorch CUDA tensor 연산과 BF16 Llama 추론 성공.
- `probe_lab.py`의 unittest **12개 통과**: 그룹 분리, MM 평행 이동 불변성, LR raw-space 동치, train-only PCA, group bootstrap, 좌/우 padding, hook 제거, 모델 상태 복구, 개입 logit 변화, attention mask/gradient.
- 00~03 notebook을 커널에서 처음부터 끝까지 실행. 04는 기본 CPU 경로 및 별도의 실제 8B 경로를 실행.
- 실제 GPU 연산 종료 후 모델 메모리 반환. 외부 포트/서비스는 추가하지 않음.

## 실제 truth pilot

Llama-2-13B, BF16, seed42, 468문장. 추출/모델 선택/평가/개입 전체 정상 종료한 실행은 약 33초, peak PyTorch GPU allocation 약 24.34 GiB. 다운로드 시간은 포함하지 않습니다. 선택은 validation으로만 했습니다.

| Probe / 선택한 block | 도시 test AUROC | 부정문 AUROC | 번역 AUROC | 수 비교 AUROC |
|---|---:|---:|---:|---:|
| MM / 14 | 1.000 | 0.899 | 1.000 | 0.983 |
| LR / 8, C=.01 | 0.998 | 0.000 | 0.992 | 0.856 |

**이 표만으로 MM이 보편적으로 우수하다고 결론내리지 않습니다.** 선택된 층도 다르고 표본도 작습니다. LR의 부정문 실패는 해당 실험의 전이 실패이며, 이를 본 뒤 부호를 뒤집어 성공으로 보고하지 않았습니다. MM 부정문의 balanced accuracy는 0.50이므로 AUROC와 threshold transfer를 구분할 실제 사례입니다.

도시 test는 24개 도시/48행입니다. group bootstrap은 고정한 모델/분류기에 조건부인 표본 불확실성만 나타냅니다. 완전 분리된 작은 평가 집합에서 percentile interval이 [1,1]이 되어도 모집단의 완전 분리를 입증하지 않습니다. seed·모델·재학습 불확실성은 별도입니다.

Additive intervention은 block14의 MM 방향과 norm을 맞춘 random 방향, 5개 사전 고정 세기, test의 6도시/12문장에서 실행했습니다. 0개입이 baseline과 동일함을 확인했습니다. `True − False` token ID는 **전체 prompt의 continuation**으로 검사했습니다. 단독 문자열 앞 공백을 tokenize하면 Llama SentencePiece가 추가 공백 token을 넣는 문제를 확인하고 수정했습니다.

## 실제 8B instruction/context pilot

Meta-Llama-3.1-8B-Instruct, 60개 fact를 먼저 60/20/20으로 나누고 paired instruction 조건을 만들었습니다. block16과 LR C=.01을 고정했습니다. original Apollo의 27개 AI-liar scenario도 평가했습니다.

| 평가 | AUROC | 주의 |
|---|---:|---|
| 본 적 있는 template, 새 fact | 1.000 | instruction/context 구분 |
| 새 template, 새 fact | 0.722 | 표현 변화에 민감 |
| AI-liar 고정 응답 / 27 scenario | 0.915 | 생성한 실제 기만 행동이 아님; threshold0 accuracy .50 |

대화 생성이나 전략적 기만 행동 검증을 하지 않았습니다. 같은 assistant content가 서로 다른 instruction 아래 놓였을 때의 분류 실험입니다. 출처·group uncertainty·원점수는 `results/deception_pilot.json`과 관련 CSV를 확인하세요. 27개 negative scenario로 1% FPR 성능을 신뢰성 있게 추정할 수 없습니다.

## Attention 부분의 검증 범위

기본 실습은 **인공 sequence state**에 sparse signal과 template shortcut을 심은 실험입니다. fact/situation split, last/mean/attention baseline, validation checkpoint 선택, held-out shortcut 반전, label shuffle, mask 불변성, planted-signal 제거를 실행했습니다. 실제 high-stakes LLM 성능을 측정한 결과로 읽으면 안 됩니다.

원본 high-stakes 공개 데이터의 config/schema를 확인했고, `labels`와 `high_stakes` 필드가 불일치하는 행을 자동으로 숨기지 않는 audit 셀을 제공합니다. 실제 high-stakes 모델 재현은 이번 기본 과정에서 제외했습니다. 라벨 규약을 먼저 정해야 하며, 모델의 기만 의도와 high-stakes request는 서로 다른 target입니다.

## 실행하지 않은 범위

- 70B 모델·원본의 전체 데이터 규모·여러 seed의 논문 재현.
- 실제 기만을 유도/관찰하는 생성 실험, 적대적 RL, monitor 회피 평가.
- 모든 층·모든 template·여러 모델 탐색, 엄밀한 causal mediation/NIE 추정.
- 실제 high-stakes 데이터에서 low-FPR 운영 성능 검증.

이 한계들은 수업의 실패가 아니라 **작은 실험의 결과를 어디까지 말할 수 있는지** 배우기 위해 명시한 경계입니다.

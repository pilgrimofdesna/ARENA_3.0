# 이력

이 장을 만들면서 내린 결정과, 새로 가져온 자료를 시간순으로 적는다. 설계 의도 전체는 `DESIGN/direction.md`에 있다.
누가 정했는지를 줄마다 밝힌다. (동준)은 동준이 정한 것, (Claude 기본안)은 동준이 아직 검토하지 않은 Claude의 선택이다. 시각은 KST.

## 결정

| 날짜 | 결정 | 누가 | 이유 |
|---|---|---|---|
| 11-19 09:52 | 1.3.1 다음 교재의 거친 방향을 만든다. 처음 판은 1.3.2 steering(다정함) | 요청 동준, 안 Claude | `DESIGN/direction.md`의 "09:52 판" |
| 11-19 10:06 | 목표를 바꾼다: "goodfire 의 뉴런을 제대로 이해하고, 그 주장의 신빙성을 검증할 수 있게 되고, 응용이 어느정도(claude의 도움 아래) 가능하게 되는 거" | 동준 | 단기 목표 |
| 11-19 10:30 | ARENA 하위 chapter 하나가 아니라 여러 곳에서 골라 붙인 5권 "뉴런을 읽기" | Claude 기본안 | 실제 LLM의 MLP 뉴런을 정면으로 다루는 ARENA chapter가 없다. `DESIGN/direction.md` "수정판" |
| 11-20 06:27 | 붙드는 특성은 정직. 다정함(호의)은 "호의에만 있는 자료를 나중에 찾아볼 것"으로만 남긴다 | 동준 | 1.3.1과 이어지고, 검증이 더 쉬워 보이고 관련 자료도 더 많을 듯 |
| 11-20 06:32 | 교재 만들기를 시작한다. direction.md를 저장소에 넣고, 새 자료를 가져올 때마다 이력에 남긴다 | 동준 | |
| 11-20 06:45 | 주 실험 모델을 Llama-3.1-8B-Instruct에서 **Qwen3-0.6B-Base**로 바꾼다. Llama-8B는 나중에 같은 측정을 다시 돌려 보는 확인용 | Claude 기본안 | 공부용 Vast 상자의 호스트가 꺼져 있다(동준 11-20). Qwen3-0.6B는 GPU 없이 CPU에서 cities 1,496문장을 약 1분에 돈다. 논문 모델(Qwen3.6-35B-A3B)과 같은 Qwen 계열이고 같은 SwiGLU MLP다. 대신 1.3.1의 Llama 숫자와 직접 비교할 수 없고, 작은 모델이라 결론이 큰 모델로 옮겨 간다는 보장이 없다. |
| 11-20 06:45 | `DESIGN/direction.md`의 "뉴런 = 읽는 방향(w_in) + 쓰는 방향(w_out)"을 고친다. SwiGLU 뉴런은 읽는 벡터가 둘(gate, up)이다 | Claude | ① Cell 02에서 식으로 보인다. direction.md 본문은 설계 당시 기록이라 그대로 둔다. |

## 가져온 자료

| 날짜 | 자료 | 출처와 판 | 라이선스 | 어디에 쓰나 | 저장소에 넣었나 |
|---|---|---|---|---|---|
| 11-20 | Qwen3-0.6B-Base 가중치 | Hugging Face `Qwen/Qwen3-0.6B-Base`, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | Apache-2.0 | 전 권의 실험 모델 | 아니요. 실행할 때 내려받는다(약 1.2 GB) |
| 11-20 | Truth-is-Universal 데이터(cities 등) | 1.3.1 한국어판 `../part31_linear_probes_ko/data/` (원 출처는 그 폴더의 `data/README.md`) | MIT | ①의 참/거짓 문장 | 복사하지 않고 `probe_lab.load_dataset`으로 읽는다 |
| 11-20 | 1.3.1 한국어판 도구 `probe_lab.py` | 같은 저장소 `../part31_linear_probes_ko/probe_lab.py` | 저장소와 같음 | split, MM probe, AUROC | 복사하지 않고 import한다 |
| 11-20 | 논문 정의와 숫자 | Tigges et al., "Neurons as an Interpretable Basis", Goodfire, 2026-11-10 (에세이 "Already Clean"의 짝 논문) | 인용만 | 활성의 정의(a = silu(gate)·up), 극(pole), 분위수 표기 p97/3, 사분면 Q1–Q4, top-320 마스킹(층마다 절댓값 기준 상위 320개만 남기고 down projection 전에 0) | 본문은 넣지 않았다. 셀에서 필요한 문장만 짧게 인용한다 |
| 11-20 | ARENA 원본 장의 아이디어 | 같은 저장소 1.1 §2(MLP), 1.2 §3(hook), 1.5.3 §2–3(뉴런 가중치 읽기, direct logit attribution, max activating examples), 1.5.4 §1·§2·§5 | 저장소와 같음 | ①②의 순서와 실습 형태 | 코드는 새로 썼다 |

## 실행 기록

| 날짜 | 무엇 | 환경 |
|---|---|---|
| 11-20 | ① 실행, 출력 저장 | 클라우드 CPU 4코어, torch 2.15.0+cpu, transformers 5.23.0, float32 |

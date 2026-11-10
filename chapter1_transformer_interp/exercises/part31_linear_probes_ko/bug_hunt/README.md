# 버그 찾기

모든 노트북은 오류 없이 끝까지 돈다. 하지만 설명이나 결론이 말하는 것과 코드가 실제로 하는 일이 다른 곳이 있다. 그곳을 찾는다.

| 노트북 | 난이도 | 위치 힌트 | 연결되는 내용 |
|---|---|---|---|
| `1_MM은_실패했나` | 쉬움 | 셀 하나 | 01 MM probe, ③ Q2(AUROC vs 정확도) |
| `2_엄격한_검정` | 쉬움~보통 | 분할~평가 | group split, 순열 귀무분포 |
| `3_가장_좋은_층` | 보통 | 없음 | 층 고르기, 귀무분포가 묻는 질문 |
| `4_정답_국가_기대` | 보통~어려움 | 함수 하나 | 02 Cell 05/06 `continuation_logprob`, A5 |

1~3은 합성 활성값(`acts/*.npz`)이고, 4는 진짜 `cities.csv`에 장난감 언어모델(`toy_lm.py`)을 쓴다. 넷 다 CPU에서 몇 초면 돈다. 커널은 `ARENA (venv-arena)`를 쓰면 된다.

## 받는 법 (인스턴스나 로컬에서)

```bash
cd "$(git rev-parse --show-toplevel)"
git fetch origin claude/project-thread-gcof9v
git checkout origin/claude/project-thread-gcof9v -- chapter1_transformer_interp/exercises/part31_linear_probes_ko/bug_hunt
```

지금 브랜치의 기록은 건드리지 않고 이 폴더만 가져온다. 노트북은 이 폴더 안에서 실행한다(경로가 상대 경로다).

## 하는 법

1. 끝까지 실행하고 결론을 읽는다.
2. 말과 실제 동작이 다른 곳을 문장으로 적는다.
3. 고치면 숫자가 어떻게 바뀔지 한 줄로 예측한다.
4. 고쳐서 돌린다. 문법이 막히면 고칠 내용을 문장으로 쓰고 `???`로 남겨 두면 된다.

정답은 저장소에 없다. Claude가 따로 가지고 있다.

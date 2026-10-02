# 데이터 출처

이 폴더의 CSV는 모두 **Truth is Universal** 저장소에서 가져왔다.

- 저장소: https://github.com/sciai-lab/Truth_is_Universal (commit `605ef00514415deb4806969a172f6e13e0798df7`)
- 라이선스: MIT (`truth_is_universal/LICENSE`)
- 논문: Lennart Bürger, Fred A. Hamprecht, Boaz Nadler, *Truth is Universal: Robust Detection of Lies in LLMs*, NeurIPS 2024. [arXiv:2407.12831](https://arxiv.org/abs/2407.12831)

저장소 README에 따르면 데이터셋은 주로 이전 논문들에서 모은 것이다.

| 파일 | 원 출처 |
|---|---|
| `cities`, `neg_cities`, `sp_en_trans`, `neg_sp_en_trans`, `larger_than`, `smaller_than`, `common_claim_true_false` | Marks & Tegmark, *The Geometry of Truth* (COLM 2024), [arXiv:2310.06824](https://arxiv.org/abs/2310.06824). `common_claim`은 그 이전의 Casper et al.에서 유래 |
| `inventors`, `animal_class`, `element_symb`, `facts` (+ `neg_*`) | Azaria & Mitchell, *The Internal State of an LLM Knows When It's Lying* (EMNLP Findings 2023)의 부분집합 |
| `cities_conj`, `cities_disj`, `cities_de`, `neg_cities_de` | Bürger et al.이 만든 결합·선택·독일어 판 |
| `real_world_scenarios/*.csv` | Bürger et al.이 Llama-3-8B-Instruct로 생성하고 사람이 분류한 역할극 응답. 시나리오는 Pacchiardi et al., *How to Catch an AI Liar* (ICLR 2024)의 goal-directed lying 시나리오에서 왔다 |

Geometry of Truth 저장소(`saprmarks/geometry-of-truth`)와 Apollo의 `deception-detection` 저장소에는 라이선스 파일이 없어 이 장에 데이터를 복사하지 않았다. ⑧의 instructed pairs는 위 `facts`의 참인 문장으로 만들었다.

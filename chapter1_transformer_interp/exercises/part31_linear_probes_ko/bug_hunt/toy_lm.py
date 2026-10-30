"""문제 4용 장난감 언어모델. Llama 대신 CPU에서 몇 초 만에 돈다.

이 파일에는 버그가 없다(문제는 노트북 쪽에 있다). 읽어도 된다.

동작 규칙:
- 토큰 = 공백으로 자른 단어. 맨 앞에 <bos>, 오른쪽에 <pad>를 붙인다.
- 위치 t의 logits는 "t 다음에 올 토큰"에 대한 점수다(진짜 LM과 같다).
- "The city of X is in" 다음에는 X의 **실제 국가** 첫 단어를 강하게 기대한다.
- 국가 이름 안에서는 다음 단어를 강하게 기대한다("the" → "Dominican" → "Republic.").
- 그 밖에는 거의 균등하다.
즉 이 모델은 "도시-국가 짝"을 안다. 참 문장의 국가 log-prob는 높고 거짓 문장은 낮아야 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "truth_is_universal" / "cities.csv"


class ToyTokenizer:
    def __init__(self, texts):
        words = sorted({w for t in texts for w in t.split()})
        self.vocab = ["<pad>", "<bos>"] + words
        self.id = {w: i for i, w in enumerate(self.vocab)}
        self.pad_id, self.bos_id = 0, 1

    def __call__(self, texts, padding: bool = False):
        """texts: str 또는 list[str]. Returns dict(input_ids, attention_mask)."""
        single = isinstance(texts, str)
        seqs = [[self.bos_id] + [self.id[w] for w in t.split()] for t in ([texts] if single else texts)]
        if single:
            return {"input_ids": seqs[0], "attention_mask": [1] * len(seqs[0])}
        T = max(map(len, seqs))
        ids = np.full((len(seqs), T), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(seqs), T), dtype=np.int64)
        for b, s in enumerate(seqs):
            ids[b, : len(s)] = s
            mask[b, : len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids):
        return [self.vocab[int(i)] for i in ids]


class ToyLM:
    """model(input_ids) → logits [B, T, V]."""

    def __init__(self, tok: ToyTokenizer, city_to_country: dict[str, str], countries, seed: int = 0):
        self.tok = tok
        V = len(tok.vocab)
        rng = np.random.default_rng(seed)
        self.base = rng.normal(scale=0.3, size=V)  # 거의 균등한 배경 점수
        ids = lambda text: tuple(tok.id[w] for w in text.split())
        self.first_word = {ids(c): tok.id[k.split()[0]] for c, k in city_to_country.items()}
        self.next_word: dict[int, set[int]] = {}
        for k in countries:
            w = k.split()
            for a, b in zip(w, w[1:]):
                self.next_word.setdefault(tok.id[a], set()).add(tok.id[b])
        self.of_id, self.is_id, self.in_id = tok.id["of"], tok.id["is"], tok.id["in"]

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        B, T = input_ids.shape
        logits = np.tile(self.base, (B, T, 1))
        for b in range(B):
            row = [int(i) for i in input_ids[b]]
            for t, cur in enumerate(row):
                if cur == self.in_id and t >= 1 and row[t - 1] == self.is_id and self.of_id in row[:t]:
                    of = t - 1 - row[:t][::-1].index(self.of_id)  # 마지막 "of"의 위치
                    city = tuple(row[of + 1 : t - 1])
                    if city in self.first_word:
                        logits[b, t, self.first_word[city]] += 8.0
                elif cur in self.next_word:
                    for nxt in self.next_word[cur]:
                        logits[b, t, nxt] += 8.0
        return logits


def load_toy():
    """Returns (cities DataFrame, tokenizer, model)."""
    df = pd.read_csv(DATA)
    c2k = {c: k + "." for c, k in zip(df["city"], df["correct_country"])}  # 문장 속 토큰은 "Russia."
    tok = ToyTokenizer(list(df["statement"]))
    model = ToyLM(tok, c2k, sorted({k + "." for k in set(df["country"]) | set(df["correct_country"])}))
    return df, tok, model


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))

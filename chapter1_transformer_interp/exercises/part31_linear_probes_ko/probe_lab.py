"""[1.3.1 한국어판] Linear probe 실습 공통 도구.

이 파일은 notebook에서 직접 구현해 보는 함수의 reference 구현과, 여러 notebook이
반복해서 쓰는 실험 도구를 담는다.

공통 규약
---------
- "layer ℓ의 activation"은 decoder block ℓ의 **출력**이다 (0-indexed).
  TransformerLens의 `blocks.{ℓ}.hook_resid_post`와 같은 위치다.
  HF `output_hidden_states`의 마지막 원소는 final norm을 거친 값이라 block 출력과 다르므로 쓰지 않는다.
- probe의 `direction`과 `bias`는 항상 **원래 activation 좌표(raw residual 좌표)** 로 저장한다.
  표준화(StandardScaler) 좌표의 계수를 그대로 residual stream에 더하면 다른 방향을 조작하게 된다.
- label 1 = 참(true), 0 = 거짓(false). probe score가 클수록 "참" 쪽이다.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

# 컨테이너의 CPU 할당량보다 BLAS thread가 많으면 4096차원 SVD가 수 분씩 걸린다.
# 무거운 선형대수는 torch(GPU)로 하고, CPU BLAS thread 수는 작게 묶어 둔다.
try:
    from threadpoolctl import threadpool_limits

    _BLAS_LIMIT = threadpool_limits(limits=min(4, os.cpu_count() or 1))
except ImportError:  # pragma: no cover
    _BLAS_LIMIT = None
torch.set_num_threads(min(4, os.cpu_count() or 1))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 경로와 기본 설정
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TIU_DIR = DATA_DIR / "truth_is_universal"
CACHE_DIR = ROOT / "cache"  # activation cache. git에 올리지 않는다.
RESULTS_DIR = ROOT / "results"  # notebook별 핵심 수치(JSON). 05·09가 모아서 읽는다.

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

# 같은 entity(도시, 단어, 사람 ...)에서 나온 문장은 같은 split에 둔다.
# 부정문 파일은 긍정문 파일과 행 단위로 정렬되어 있으므로 같은 group 규칙을 쓴다.
_GROUP_PATTERNS: dict[str, str] = {
    "sp_en_trans": r"The Spanish word '([^']+)'",
    "inventors": r"^(.*?) (?:lived|did not live) in",
    "animal_class": r"^The (.*?) is (?:not )?an? ",
    "element_symb": r"^(.*?) (?:has|does not have) the symbol",
}


def _topic(name: str) -> str:
    """`neg_cities` → `cities`, `cities_de` → `cities` 처럼 파생 데이터의 주제 이름을 돌려준다."""
    base = name.removeprefix("neg_")
    return base.removesuffix("_de")


def load_dataset(name: str) -> pd.DataFrame:
    """Truth-is-Universal 데이터셋 하나를 읽고 `group`, `dataset`, `polarity` 열을 붙인다.

    Returns: columns ⊇ [statement, label, group, dataset, polarity].
      - group: split 단위. 같은 entity에서 나온 문장은 같은 group 문자열을 가진다.
      - polarity: 부정문(`neg_*`)이면 -1, 아니면 +1.
    """
    path = TIU_DIR / f"{name}.csv"
    if not path.exists():
        path = TIU_DIR / "real_world_scenarios" / f"{name}.csv"
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(int)
    topic = _topic(name)
    if "city" in df.columns:
        groups = df["city"].astype(str)
    elif {"n1", "n2"} <= set(df.columns):
        lo = np.minimum(df["n1"], df["n2"])
        hi = np.maximum(df["n1"], df["n2"])
        groups = pd.Series([f"{a}-{b}" for a, b in zip(lo, hi)])
    elif topic in _GROUP_PATTERNS:
        pattern = re.compile(_GROUP_PATTERNS[topic])
        groups = df["statement"].map(
            lambda s: (m.group(1).lower() if (m := pattern.search(s)) else None)
        )
        # 패턴이 안 맞는 행은 자기 자신을 group으로 쓴다(누수 방향으로 실패하지 않도록 경고).
        missing = groups.isna()
        if missing.any():
            groups[missing] = [f"row{i}" for i in np.flatnonzero(missing)]
    else:
        groups = pd.Series([f"row{i}" for i in range(len(df))])
    df["group"] = [f"{topic}:{g}" for g in groups]
    df["dataset"] = name
    df["polarity"] = -1 if name.startswith("neg_") else 1
    return df


def load_datasets(names: Sequence[str]) -> pd.DataFrame:
    """여러 데이터셋을 이어 붙인다. index는 0..N-1로 다시 매긴다."""
    return pd.concat([load_dataset(n) for n in names], ignore_index=True)


@dataclass(frozen=True)
class Split:
    """group 단위로 나눈 행 index 배열."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def group_split(
    groups: Sequence[str] | np.ndarray,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> Split:
    """같은 group의 행이 서로 다른 split에 들어가지 않도록 나눈다.

    groups: [N] 각 행의 group 이름.
    Returns: train/val/test 행 index (각각 오름차순 정렬).
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(round(fractions[0] * len(unique)))
    n_val = int(round(fractions[1] * len(unique)))
    assigned = {
        "train": set(unique[:n_train]),
        "val": set(unique[n_train : n_train + n_val]),
        "test": set(unique[n_train + n_val :]),
    }
    idx = {k: np.flatnonzero(np.isin(groups, list(v))) for k, v in assigned.items()}
    return Split(idx["train"], idx["val"], idx["test"])


# ---------------------------------------------------------------------------
# 모델과 activation 추출
# ---------------------------------------------------------------------------


def load_model(
    model_id: str = DEFAULT_MODEL,
    dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
):
    """HF causal LM과 tokenizer를 불러온다. 평가 모드, right padding.

    Returns: (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    from importlib.metadata import version as _version

    from packaging.version import Version

    # transformers 4.56부터 `torch_dtype` 대신 `dtype` 인자를 쓴다.
    dtype_kw = "dtype" if Version(_version("transformers")) >= Version("4.56") else "torch_dtype"
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()  # notebook 출력에 weight 로딩 진행 막대를 남기지 않는다.
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **{dtype_kw: dtype})
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def random_init_model(model_id: str = DEFAULT_MODEL, seed: int = 0, dtype: torch.dtype = torch.bfloat16):
    """같은 구조(config)에 **무작위 초기화 weight**를 넣은 대조군 모델. tokenizer는 원래 것을 쓴다."""
    from importlib.metadata import version as _version

    from packaging.version import Version
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id)
    dtype_kw = "dtype" if Version(_version("transformers")) >= Version("4.56") else "torch_dtype"
    torch.manual_seed(seed)
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        model = AutoModelForCausalLM.from_config(cfg, **{dtype_kw: dtype})
    model.eval()
    model.requires_grad_(False)
    return model


def decoder_layers(model) -> torch.nn.ModuleList:
    """Llama/Qwen/Mistral 계열의 decoder block 목록."""
    return model.model.layers


def _block_output(out) -> Tensor:
    """decoder block forward의 반환값에서 residual stream tensor [B, S, D]를 꺼낸다.

    transformers 5.x의 Llama block은 Tensor를, 4.x 일부 버전은 (Tensor, ...) tuple을 돌려준다.
    """
    return out[0] if isinstance(out, tuple) else out


def token_positions(
    attention_mask: Tensor, offset: int = 0
) -> Tensor:
    """right padding batch에서 각 행의 마지막 유효 token 위치 + offset.

    attention_mask: [B, S] (1 = 유효 token).
    offset: 0이면 마지막 token, -1이면 그 직전 token.
    Returns: [B] long tensor.
    """
    last = attention_mask.sum(dim=1) - 1
    return last + offset


@torch.inference_mode()
def extract_activations(
    model,
    tokenizer,
    texts: Sequence[str],
    layers: Sequence[int],
    *,
    batch_size: int = 32,
    token_offset: int = 0,
    char_spans: Sequence[tuple[int, int]] | None = None,
    add_special_tokens: bool = True,
    show_progress: bool = False,
) -> dict[int, Tensor]:
    """각 text의 지정 위치에서 decoder block 출력(resid_post)을 모은다.

    - char_spans가 None이면 마지막 유효 token(+token_offset) 하나를 읽는다.
    - char_spans[i] = (start, end)이면 그 문자 구간에 걸친 token들의 **평균**을 읽는다
      (예: chat 대화에서 assistant 답변 부분만 평균).
    - chat template로 이미 BOS가 들어간 문자열은 add_special_tokens=False로 넘긴다.

    Returns: {layer: [N, D] float32 CPU tensor}
    """
    layers = list(layers)
    blocks = decoder_layers(model)
    device = next(model.parameters()).device
    out: dict[int, list[Tensor]] = {l: [] for l in layers}
    current: dict[str, Tensor] = {}

    def make_hook(layer: int):
        def hook(module, inputs, output):
            h = _block_output(output)  # [B, S, D]
            if "pool" in current:
                w = current["pool"]  # [B, S] 평균 가중치
                v = (w.unsqueeze(-1) * h.float()).sum(1)
            else:
                b = torch.arange(h.shape[0], device=h.device)
                v = h[b, current["pos"]].float()
            out[layer].append(v.cpu())

        return hook

    handles = [blocks[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        starts = range(0, len(texts), batch_size)
        iterator: Iterable[int] = starts
        if show_progress:
            from tqdm.auto import tqdm

            iterator = tqdm(starts, desc="activations")
        for s in iterator:
            batch = list(texts[s : s + batch_size])
            if char_spans is None:
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=add_special_tokens,
                ).to(device)
                current.pop("pool", None)
                current["pos"] = token_positions(enc["attention_mask"], token_offset)
            else:
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    return_offsets_mapping=True,
                    add_special_tokens=add_special_tokens,
                )
                offsets = enc.pop("offset_mapping")  # [B, S, 2]
                spans = torch.tensor(char_spans[s : s + batch_size])  # [B, 2]
                inside = (offsets[..., 0] >= spans[:, :1]) & (offsets[..., 1] <= spans[:, 1:])
                inside &= offsets[..., 1] > offsets[..., 0]  # BOS 같은 빈 token 제외
                inside &= enc["attention_mask"].bool()
                if (inside.sum(1) == 0).any():
                    raise ValueError("char span 안에 token이 없는 text가 있다.")
                w = inside.float() / inside.sum(1, keepdim=True)
                current["pool"] = w.to(device)
                enc = enc.to(device)
            # lm_head(128k vocab)는 필요 없으므로 base model만 실행한다.
            model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        for h in handles:
            h.remove()
    return {l: torch.cat(v) for l, v in out.items()}


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__")


class ActivationStore:
    """데이터셋 이름 단위로 activation을 계산하고 disk cache에 저장한다.

    cache 파일: cache/<model>/<dataset>__<tag>__L<layer>.pt  (float16, [N, D])
    모델은 처음 필요할 때만 불러온다. cache가 모두 있으면 GPU 없이도 분석할 수 있다.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, model=None, tokenizer=None):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.dir = CACHE_DIR / _slug(model_id)
        self.dir.mkdir(parents=True, exist_ok=True)

    def ensure_model(self):
        if self.model is None:
            self.model, self.tokenizer = load_model(self.model_id)
        return self.model, self.tokenizer

    def _path(self, key: str, tag: str, layer: int) -> Path:
        return self.dir / f"{key}__{tag}__L{layer}.pt"

    def get(
        self,
        key: str,
        texts: Sequence[str],
        layers: Sequence[int],
        *,
        tag: str = "last",
        token_offset: int = 0,
        char_spans: Sequence[tuple[int, int]] | None = None,
        add_special_tokens: bool = True,
        batch_size: int = 32,
    ) -> dict[int, np.ndarray]:
        """`key`라는 이름으로 texts의 activation을 돌려준다(없으면 계산 후 저장).

        같은 key·tag에 다른 texts를 넘기면 잘못된 cache를 읽으므로 key를 내용에 맞게 짓는다.
        Returns: {layer: [N, D] float32 numpy}
        """
        # 같은 key에 다른 texts가 들어오면 낡은 cache를 조용히 읽지 않도록 내용 hash를 함께 저장한다.
        import hashlib

        digest = hashlib.sha1("\x1e".join(texts).encode()).hexdigest()[:16]
        stamp = self.dir / f"{key}__{tag}.sha1"
        if stamp.exists() and stamp.read_text() != digest:
            for old in self.dir.glob(f"{key}__{tag}__L*.pt"):
                old.unlink()
        stamp.write_text(digest)
        missing = [l for l in layers if not self._path(key, tag, l).exists()]
        if missing:
            model, tok = self.ensure_model()
            acts = extract_activations(
                model,
                tok,
                texts,
                missing,
                batch_size=batch_size,
                token_offset=token_offset,
                char_spans=char_spans,
                add_special_tokens=add_special_tokens,
            )
            for l, a in acts.items():
                torch.save(a.to(torch.float16), self._path(key, tag, l))
        result = {}
        for l in layers:
            a = torch.load(self._path(key, tag, l))
            if a.shape[0] != len(texts):
                raise ValueError(f"cache {key} L{l}의 행 수가 texts와 다르다. cache를 지우고 다시 계산하라.")
            result[l] = a.float().numpy()
        return result

    def dataset(
        self, name: str, layers: Sequence[int], *, token_offset: int = 0
    ) -> dict[int, np.ndarray]:
        """Truth-is-Universal 데이터셋의 statement 마지막 token activation."""
        df = load_dataset(name)
        tag = "last" if token_offset == 0 else f"off{token_offset}"
        return self.get(name, df["statement"].tolist(), layers, tag=tag, token_offset=token_offset)


def render_chat(tokenizer, messages: list[dict[str, str]]) -> tuple[str, tuple[int, int]]:
    """chat template로 렌더링한 문자열과 마지막 메시지 content의 문자 구간 (start, end).

    마지막 메시지는 보통 assistant 답변이다. template가 붙이는 header·<|eot_id|>는 구간에서 빠진다.
    렌더링된 문자열에는 이미 BOS가 있으므로 tokenize할 때 add_special_tokens=False를 쓴다.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    content = messages[-1]["content"]
    start = text.rfind(content)
    if start < 0:
        raise ValueError("렌더링된 대화에서 마지막 메시지를 찾지 못했다.")
    return text, (start, start + len(content))


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


@dataclass
class LinearProbe:
    """score(x) = x · direction + bias. direction/bias는 raw activation 좌표다."""

    direction: np.ndarray  # [D]
    bias: float
    kind: str
    info: dict = field(default_factory=dict)

    def score(self, X: np.ndarray) -> np.ndarray:
        """X: [N, D] → [N] (양수 = 참 쪽)."""
        return X @ self.direction + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) > 0).astype(int)

    @property
    def unit(self) -> np.ndarray:
        """단위 길이 방향 [D]."""
        return self.direction / np.linalg.norm(self.direction)


def fit_mass_mean(X: np.ndarray, y: np.ndarray) -> LinearProbe:
    """Difference-of-means(mass-mean) probe.

    θ = μ₁ − μ₀,  bias = −θ·(μ₁ + μ₀)/2  → 결정 경계가 두 평균의 중점을 지난다.
    bias를 빼고 `x·θ > 0`으로 판정하면 residual stream 원점에 의존하는 다른 분류기가 된다.
    """
    mu1 = X[y == 1].mean(0)
    mu0 = X[y == 0].mean(0)
    theta = mu1 - mu0
    return LinearProbe(theta, float(-theta @ (mu1 + mu0) / 2), "MM")


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 0.1,
    max_iter: int = 200,
    device: str | None = None,
) -> LinearProbe:
    """L2 정규화 logistic regression probe.

    목적함수는 sklearn `LogisticRegression(C=C)`와 같다:
        minimize  ½‖w‖² + C · Σᵢ log-loss(zᵢ·w + b, yᵢ)     (bias는 정규화하지 않음)
    표준화 좌표 z = (x − m)/s 에서 학습한 뒤 raw 좌표로 되돌린다:
        w_raw = w_z / s,   b_raw = b_z − w_raw · m
    C가 작을수록 강한 정규화. n < D(표본 수 < 차원)이면 거의 항상 선형 분리 가능하므로
    정규화가 없으면 해가 발산한다.

    sklearn의 CPU L-BFGS는 D=4096, n≈900에서 수 분이 걸려 torch L-BFGS로 같은 문제를 푼다.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = X.mean(0)
    s = X.std(0) + 1e-6
    Z = torch.tensor((X - m) / s, dtype=torch.float32, device=device)
    t = torch.tensor(y, dtype=torch.float32, device=device)
    w = torch.zeros(Z.shape[1], device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.LBFGS(
        [w, b],
        lr=1.0,
        max_iter=max_iter,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-10,
    )

    def closure():
        opt.zero_grad()
        logits = Z @ w + b
        loss = C * torch.nn.functional.binary_cross_entropy_with_logits(
            logits, t, reduction="sum"
        ) + 0.5 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    w_z = w.detach().cpu().double().numpy()
    w_raw = w_z / s
    b_raw = float(b.detach().cpu().item() - w_raw @ m)
    return LinearProbe(w_raw, b_raw, "LR", {"C": C})


def fit_probe(kind: str, X: np.ndarray, y: np.ndarray, **kw) -> LinearProbe:
    if kind == "MM":
        return fit_mass_mean(X, y)
    if kind == "LR":
        return fit_logistic(X, y, **kw)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# 지표와 통계
# ---------------------------------------------------------------------------


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """순위 기반 분리도. threshold와 무관하다. 0.5 = 무작위, 0 = 순위가 완전히 뒤집힘."""
    return float(roc_auc_score(y, score))


def accuracy(y: np.ndarray, score: np.ndarray, threshold: float = 0.0) -> float:
    """score > threshold 를 참으로 판정했을 때의 정확도."""
    return float(np.mean((score > threshold).astype(int) == y))


def tpr_at_fpr(y_pos_is_1: np.ndarray, score: np.ndarray, fpr: float = 0.01) -> tuple[float, float]:
    """음성(0) score 분포의 (1−fpr) 분위수를 threshold로 삼았을 때의 TPR.

    Returns: (tpr, threshold). 음성 표본이 1/fpr개보다 적으면 그 fpr은 경험적으로 추정할 수 없다.
    """
    neg = score[y_pos_is_1 == 0]
    pos = score[y_pos_is_1 == 1]
    thr = float(np.quantile(neg, 1 - fpr, method="higher"))
    return float(np.mean(pos > thr)), thr


def evaluate(probe: LinearProbe, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    s = probe.score(X)
    return {"auroc": auroc(y, s), "acc": accuracy(y, s)}


def permutation_null(
    fit: Callable[[np.ndarray, np.ndarray], LinearProbe],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_perm: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """train label을 섞어 다시 학습했을 때 test AUROC의 분포(귀무분포).

    test label은 섞지 않는다. 묻는 질문: "label이 activation과 무관하다면
    이 학습 절차가 이만한 test AUROC를 얼마나 자주 우연히 내는가?"
    Returns: [n_perm] AUROC 배열.
    """
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        probe = fit(X_train, rng.permutation(y_train))
        null[i] = auroc(y_test, probe.score(X_test))
    return null


def permutation_p_value(null: np.ndarray, observed: float) -> float:
    """단측 p = (1 + #{null ≥ observed}) / (1 + n_perm). 0이 나오지 않도록 +1 보정."""
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def group_bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = auroc,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """group 단위로 재표집한 percentile 신뢰구간.

    고정된 probe에 대한 **평가 표본** 변동만 반영한다(재학습·seed·층 선택의 불확실성은 미포함).
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], score[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def dprime(probe: LinearProbe, X: np.ndarray, y: np.ndarray) -> float:
    """probe 방향 위에서 두 class 평균이 class 내부 표준편차의 몇 배 떨어져 있는가.

    AUROC가 1.0으로 포화된 뒤에도 분리의 '여유'를 비교할 수 있는 지표다.
    """
    s = X @ probe.unit
    a, b = s[y == 1], s[y == 0]
    return float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var())))


def fit_truth_polarity(
    X_by_dataset: dict[str, np.ndarray],
    y_by_dataset: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Bürger et al. (2024)의 일반 진실 방향 t_G와 극성 의존 방향 t_P.

    각 데이터셋을 자기 평균으로 중심화한 뒤 다음 선형 모형을 최소제곱으로 푼다:
        x̃ ≈ τ · t_G + τ·p · t_P,   τ = ±1(참/거짓), p = +1(긍정문) / −1(부정문)
    데이터셋 이름이 `neg_`로 시작하면 부정문으로 본다.
    Returns: (t_G [D], t_P [D])
    """
    rows, coef = [], []
    for name, X in X_by_dataset.items():
        tau = 2.0 * y_by_dataset[name] - 1.0
        pol = -1.0 if name.startswith("neg_") else 1.0
        rows.append(X - X.mean(0))
        coef.append(np.stack([tau, tau * pol], axis=1))
    Xc = torch.tensor(np.concatenate(rows), dtype=torch.float64)
    A = torch.tensor(np.concatenate(coef), dtype=torch.float64)
    T = torch.linalg.lstsq(A, Xc).solution  # [2, D]
    return T[0].numpy(), T[1].numpy()


@dataclass
class PCA:
    """train 데이터에만 맞춘 PCA. components: [k, D] (행이 주성분), explained: [k] 분산 비율."""

    mean: np.ndarray
    components: np.ndarray
    explained: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: [N, D] → [N, k]"""
        return (X - self.mean) @ self.components.T


def fit_pca(X: np.ndarray, k: int = 10) -> PCA:
    """중심화한 X의 thin SVD로 상위 k개 주성분을 구한다(GPU가 있으면 GPU에서)."""
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    mu = Xt.mean(0)
    _, S, Vh = torch.linalg.svd(Xt - mu, full_matrices=False)
    var = S**2
    return PCA(
        mu.cpu().numpy(),
        Vh[:k].cpu().numpy(),
        (var[:k] / var.sum()).cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# 행동 측정과 개입
# ---------------------------------------------------------------------------

# few-shot 예시는 이 장의 어느 데이터셋에도 없는 도시·단어로 골랐다(예시를 복사해 맞히는 경로 차단).
FEW_SHOT_PROMPT = (
    "The city of Bergen is in Norway. This statement is: TRUE\n"
    "The city of Graz is in Brazil. This statement is: FALSE\n"
    "The Spanish word 'queso' means 'cheese'. This statement is: TRUE\n"
    "The Spanish word 'espejo' means 'rain'. This statement is: FALSE\n"
)
SUFFIX = " This statement is:"


def answer_token_ids(tokenizer, answers: tuple[str, str] = (" TRUE", " FALSE")) -> tuple[int, int]:
    """답변 문자열이 prompt 뒤에서 **한 token**으로 이어지는지 확인하고 그 id를 돌려준다."""
    ids = []
    for a in answers:
        toks = tokenizer(a, add_special_tokens=False)["input_ids"]
        if len(toks) != 1:
            raise ValueError(f"{a!r}가 한 token이 아니다: {toks}")
        ids.append(toks[0])
    return ids[0], ids[1]


def build_prompts(statements: Sequence[str], prefix: str = FEW_SHOT_PROMPT) -> list[str]:
    return [prefix + s + SUFFIX for s in statements]


@contextlib.contextmanager
def add_to_residual(
    model,
    vectors: dict[int, Tensor],
    positions: Callable[[Tensor], Tensor] | None = None,
) -> Iterator[None]:
    """`with` 블록 안에서 block ℓ 출력에 vectors[ℓ]를 더한다.

    vectors: {layer: [D] tensor}
    positions: hook이 받은 hidden [B, S, D]를 받아 더할 위치의 bool mask [B, S]를 돌려주는 함수.
               None이면 모든 위치에 더한다.
    블록을 나가면 hook은 항상 제거된다.
    """
    blocks = decoder_layers(model)
    handles = []

    def make_hook(vec: Tensor):
        def hook(module, inputs, output):
            h = _block_output(output)
            v = vec.to(h.dtype).to(h.device)
            if positions is None:
                h = h + v
            else:
                mask = positions(h).to(h.device).unsqueeze(-1).to(h.dtype)  # [B, S, 1]
                h = h + mask * v
            if isinstance(output, tuple):
                return (h,) + tuple(output[1:])
            return h

        return hook

    try:
        for l, vec in vectors.items():
            handles.append(blocks[l].register_forward_hook(make_hook(vec)))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.inference_mode()
def true_false_margin(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int = 32,
    vectors: dict[int, Tensor] | None = None,
    positions_from_end: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """마지막 위치의 다음 token 분포에서 P(TRUE) − P(FALSE)와 logit 차이를 잰다.

    vectors가 있으면 각 prompt의 '끝에서 k번째' token들(positions_from_end, 예: suffix 직전의
    마침표와 그 앞 token)에 더한 상태로 측정한다. right padding이므로 위치를 행마다 계산한다.
    Returns: {"p_diff": [N], "logit_diff": [N], "p_sum": [N]}
    """
    t_id, f_id = answer_token_ids(tokenizer)
    device = next(model.parameters()).device
    p_diff, logit_diff, p_sum = [], [], []
    for s in range(0, len(prompts), batch_size):
        enc = tokenizer(list(prompts[s : s + batch_size]), return_tensors="pt", padding=True).to(device)
        last = enc["attention_mask"].sum(1) - 1  # [B]
        ctx = contextlib.nullcontext()
        if vectors:
            assert positions_from_end is not None
            ks = torch.tensor(list(positions_from_end), device=device)

            def positions(h: Tensor, last=last, ks=ks) -> Tensor:
                S = h.shape[1]
                target = last.unsqueeze(1) - ks.unsqueeze(0)  # [B, K]
                mask = torch.zeros(h.shape[0], S, dtype=torch.bool, device=h.device)
                mask.scatter_(1, target, True)
                return mask

            ctx = add_to_residual(model, vectors, positions)
        with ctx:
            logits = model(**enc).logits  # [B, S, V]
        b = torch.arange(logits.shape[0], device=device)
        final = logits[b, last].float()  # [B, V]
        probs = final.softmax(-1)
        p_diff.append((probs[:, t_id] - probs[:, f_id]).cpu())
        p_sum.append((probs[:, t_id] + probs[:, f_id]).cpu())
        logit_diff.append((final[:, t_id] - final[:, f_id]).cpu())
    return {
        "p_diff": torch.cat(p_diff).numpy(),
        "logit_diff": torch.cat(logit_diff).numpy(),
        "p_sum": torch.cat(p_sum).numpy(),
    }


@contextlib.contextmanager
def edit_residual(
    model,
    fns: dict[tuple[int, int], Callable[[Tensor], Tensor]],
    last_index: Callable[[], Tensor],
) -> Iterator[None]:
    """`with` 블록 안에서 block 출력의 특정 위치 벡터를 함수로 바꾼다.

    fns[(layer, k)] = f : [B, D] float32 → [B, D]. k는 '끝에서 k번째' 위치(0 = 마지막 token).
    last_index(): 현재 batch 각 행의 마지막 유효 token 위치 [B]를 돌려주는 함수
                  (forward 직전에 바깥에서 갱신한다).
    """
    blocks = decoder_layers(model)
    by_layer: dict[int, list[tuple[int, Callable]]] = {}
    for (l, k), f in fns.items():
        by_layer.setdefault(l, []).append((k, f))
    handles = []

    def make_hook(items):
        def hook(module, inputs, output):
            h = _block_output(output).clone()
            b = torch.arange(h.shape[0], device=h.device)
            last = last_index().to(h.device)
            for k, f in items:
                pos = last - k
                h[b, pos] = f(h[b, pos].float()).to(h.dtype)
            if isinstance(output, tuple):
                return (h,) + tuple(output[1:])
            return h

        return hook

    try:
        for l, items in by_layer.items():
            handles.append(blocks[l].register_forward_hook(make_hook(items)))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.inference_mode()
def context_activations(
    model,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    positions_from_end: Sequence[int],
    batch_size: int = 32,
) -> dict[tuple[int, int], np.ndarray]:
    """few-shot prompt 안에서 '끝에서 k번째' 위치들의 block 출력을 한 번에 모은다.

    Returns: {(layer, k): [N, D] float32 numpy}
    """
    blocks = decoder_layers(model)
    out: dict[tuple[int, int], list[Tensor]] = {(l, k): [] for l in layers for k in positions_from_end}
    state: dict[str, Tensor] = {}

    def make_hook(l):
        def hook(module, inputs, output):
            h = _block_output(output)
            b = torch.arange(h.shape[0], device=h.device)
            for k in positions_from_end:
                out[(l, k)].append(h[b, state["last"] - k].float().cpu())

        return hook

    handles = [blocks[l].register_forward_hook(make_hook(l)) for l in layers]
    device = next(model.parameters()).device
    try:
        for s in range(0, len(prompts), batch_size):
            enc = tokenizer(list(prompts[s : s + batch_size]), return_tensors="pt", padding=True).to(device)
            state["last"] = enc["attention_mask"].sum(1) - 1
            model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        for h in handles:
            h.remove()
    return {key: torch.cat(v).numpy() for key, v in out.items()}


def suffix_positions(tokenizer, suffix: str = SUFFIX) -> tuple[int, int]:
    """prompt 끝에서 셌을 때 (statement의 마지막 단어 token, 마침표 token)의 위치.

    prompt = ... "<statement>." + " This statement is:" 이므로, suffix token 수를 k라 하면
    마침표는 끝에서 k번째(0-indexed 기준 last-k), 그 앞 token은 last-k-1이다.
    Returns: (k+1, k)  → true_false_margin의 positions_from_end에 그대로 넣는다.
    """
    k = len(tokenizer(suffix, add_special_tokens=False)["input_ids"])
    return (k + 1, k)


# ---------------------------------------------------------------------------
# 개념 제거(concept erasure)
# ---------------------------------------------------------------------------


@dataclass
class LeaceEraser:
    """LEACE(Belrose et al., 2023)의 이진 label용 닫힌 형태.

    x' = x − Wᵖ P W (x − μ)  (W: whitening, P: W Σ_xz 열공간으로의 정사영)
    erase 후에는 어떤 linear classifier도 label을 μ 차이로 읽을 수 없다(학습 분포 기준).
    """

    mean: np.ndarray  # [D]
    proj: np.ndarray  # [D, D]  x − mean 에 곱하는 제거 행렬 (Wᵖ P W)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return X - (X - self.mean) @ self.proj.T


def fit_leace(X: np.ndarray, y: np.ndarray, eps: float = 1e-4) -> LeaceEraser:
    """X: [N, D], y: [N] ∈ {0,1}. D×D 고유분해를 하므로 D=4096이면 GPU에서 1초 남짓 걸린다.

    GPU에서는 float32로 계산한다(eps보다 작은 고유값은 버리므로 정밀도 문제가 작다).
    """
    dtype = torch.float32 if DEVICE == "cuda" else torch.float64
    Xt = torch.tensor(X, dtype=dtype, device=DEVICE)
    yt = torch.tensor(y, dtype=dtype, device=DEVICE)
    mu = Xt.mean(0)
    Xc = Xt - mu
    sigma = Xc.T @ Xc / len(Xt)
    sigma_xz = Xc.T @ (yt - yt.mean()) / len(Xt)  # [D]
    evals, evecs = torch.linalg.eigh(sigma)
    keep = evals > eps * evals.max()
    inv_sqrt = (evecs[:, keep] / evals[keep].sqrt()) @ evecs[:, keep].T  # W
    sqrt = (evecs[:, keep] * evals[keep].sqrt()) @ evecs[:, keep].T  # W⁺
    u = inv_sqrt @ sigma_xz
    u = u / u.norm()
    P = torch.outer(u, u)
    proj = sqrt @ P @ inv_sqrt
    return LeaceEraser(mu.cpu().numpy(), proj.cpu().numpy())


# ---------------------------------------------------------------------------
# 그림 규약 (matplotlib)
#   색은 역할에 고정한다: 참 = 파랑, 거짓 = 주황, 세 번째 계열 = 청록.
#   AUROC 행렬은 0.5(우연)를 회색 중점으로 둔 파랑↔빨강 diverging map을 쓴다.
#   (0.5 아래 = 순위가 뒤집힘, 위 = 맞는 방향). 그림 안의 글자는 영어로 둔다(CJK 글꼴 의존 방지).
# ---------------------------------------------------------------------------

C_TRUE = "#2a78d6"
C_FALSE = "#eb6834"
C_THIRD = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def use_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "font.size": 10,
            "legend.frameon": False,
            "legend.labelcolor": INK2,
            "lines.linewidth": 2.0,
            "figure.dpi": 110,
        }
    )


def auroc_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("auroc", ["#e34948", "#f0efec", "#2a78d6"])


def auroc_heatmap(ax, M: np.ndarray, rows: Sequence[str], cols: Sequence[str], title: str = "") -> None:
    """AUROC 표를 0.5 중심 diverging 색으로 그리고 칸마다 값을 쓴다."""
    im = ax.imshow(M, cmap=auroc_cmap(), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v - 0.5) > 0.38 else INK)
    ax.set_title(title)
    return im


def scatter_classes(
    ax,
    Z: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray | None = None,
    title: str = "",
    xlabel: str = "PC1",
    ylabel: str = "PC2",
) -> None:
    """2차원 투영을 참(파랑)/거짓(주황)으로, 부정문은 x 표식으로 그린다."""
    pol = np.ones(len(y)) if polarity is None else polarity
    for lab, color, name in [(1, C_TRUE, "true"), (0, C_FALSE, "false")]:
        for p, marker, suffix in [(1, "o", ""), (-1, "x", " (negated)")]:
            m = (y == lab) & (pol == p)
            if m.any():
                ax.scatter(Z[m, 0], Z[m, 1], s=14, c=color, marker=marker, alpha=0.65,
                           linewidths=1.0 if marker == "x" else 0.4,
                           edgecolors=SURFACE if marker == "o" else None, label=name + suffix)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="best", fontsize=8, markerscale=1.5)


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------


def _to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _to_jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def save_results(name: str, results: dict) -> Path:
    """results/<name>.json 에 핵심 수치를 저장한다(05, 09 notebook이 모아서 읽는다)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    payload = {"saved_at": time.strftime("%Y-%m-%d %H:%M"), **_to_jsonable(results)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return path


def load_results(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())

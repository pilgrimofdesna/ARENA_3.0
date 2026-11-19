"""[뉴런을 읽기] 공통 도구.

1.3.1 한국어판의 `probe_lab.py`(데이터, split, probe, AUROC)를 그대로 가져다 쓰고,
여기에는 SwiGLU MLP 뉴런을 꺼내고, 끄고, 가중치를 읽는 도구만 둔다.

공통 규약
---------
- 층 ℓ의 MLP는 Qwen3/Llama 계열의 SwiGLU다. 입력 x̂는 attention 뒤 RMSNorm(`post_attention_layernorm`)의 출력이다.
    g = W_gate x̂,  u = W_up x̂,  a = SiLU(g) ⊙ u,  MLP(x̂) = W_down a
  뉴런 i의 **활성**은 a_i = SiLU(g_i)·u_i (논문의 정의와 같다. 이 모델은 MoE가 아니라 router weight가 없다).
  뉴런 i가 residual에 **쓰는 것**은 a_i · W_down[:, i].
- 활성은 `down_proj`의 **입력**에서 읽는다. `act_fn`의 출력(SiLU(g)만)이나 `up_proj`의 출력(u만)은 a가 아니다.
- "층 ℓ의 residual"은 1.3.1과 같이 decoder block ℓ의 출력이다(0부터 센다).
- 문장 하나에서 읽는 위치는 마지막 token(cities 문장에서는 마침표)이다.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parent
PROBE_DIR = ROOT.parent / "part31_linear_probes_ko"
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))
import probe_lab as pl  # noqa: E402  (1.3.1의 데이터·probe 도구)

CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
DEFAULT_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def load_model(model_id: str = DEFAULT_MODEL, revision: str | None = DEFAULT_REVISION, dtype=torch.float32):
    """CPU에서도 돌도록 float32로 불러온다. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype).to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def layers(model):
    return model.model.layers


def mlp_weights(model, layer: int) -> dict[str, np.ndarray]:
    """층 하나의 MLP 가중치. W_gate, W_up: [F, D] (행 i = 뉴런 i가 읽는 벡터), W_down: [D, F] (열 i = 뉴런 i가 쓰는 벡터)."""
    mlp = layers(model)[layer].mlp
    return {
        "W_gate": mlp.gate_proj.weight.detach().float().cpu().numpy(),
        "W_up": mlp.up_proj.weight.detach().float().cpu().numpy(),
        "W_down": mlp.down_proj.weight.detach().float().cpu().numpy(),
    }


def silu(x):
    return x / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# 활성 꺼내기
# ---------------------------------------------------------------------------
_KINDS = ("a", "g", "u", "xhat", "resid")


@torch.inference_mode()
def extract(
    model,
    tok,
    texts: Sequence[str],
    layer_ids: Sequence[int],
    kinds: Sequence[str] = ("a", "resid"),
    *,
    batch_size: int = 64,
    use_cache: bool = True,
) -> dict[str, dict[int, np.ndarray]]:
    """각 문장의 마지막 token에서 값을 모은다.

    kinds: "a"(뉴런 활성, down_proj 입력), "g"(gate_proj 출력), "u"(up_proj 출력),
           "xhat"(MLP 입력 = post_attention_layernorm 출력), "resid"(block 출력).
    Returns: {kind: {layer: [N, dim] float32}}
    """
    layer_ids, kinds = list(layer_ids), list(kinds)
    assert set(kinds) <= set(_KINDS), kinds
    key = hashlib.sha1(
        json.dumps([model.config._name_or_path, list(texts), layer_ids, kinds]).encode()
    ).hexdigest()[:16]
    path = CACHE_DIR / f"extract_{key}.npz"
    if use_cache and path.exists():
        z = np.load(path)
        return {k: {l: z[f"{k}_{l}"].astype(np.float32) for l in layer_ids} for k in kinds}

    store = {k: {l: [] for l in layer_ids} for k in kinds}
    cur: dict[str, Tensor] = {}

    def take(kind, layer):
        def put(t):
            b = torch.arange(t.shape[0], device=t.device)
            store[kind][layer].append(t[b, cur["pos"]].float().cpu())
        return put

    handles = []
    for l in layer_ids:
        blk = layers(model)[l]
        mlp = blk.mlp
        if "a" in kinds:
            p = take("a", l)
            handles.append(mlp.down_proj.register_forward_pre_hook(lambda m, inp, p=p: p(inp[0])))
        if "g" in kinds:
            p = take("g", l)
            handles.append(mlp.gate_proj.register_forward_hook(lambda m, i, o, p=p: p(o)))
        if "u" in kinds:
            p = take("u", l)
            handles.append(mlp.up_proj.register_forward_hook(lambda m, i, o, p=p: p(o)))
        if "xhat" in kinds:
            p = take("xhat", l)
            handles.append(mlp.register_forward_pre_hook(lambda m, inp, p=p: p(inp[0])))
        if "resid" in kinds:
            p = take("resid", l)
            handles.append(blk.register_forward_hook(lambda m, i, o, p=p: p(o[0] if isinstance(o, tuple) else o)))
    device = next(model.parameters()).device
    try:
        for s in range(0, len(texts), batch_size):
            enc = tok(list(texts[s : s + batch_size]), return_tensors="pt", padding=True).to(device)
            cur["pos"] = enc["attention_mask"].sum(1) - 1
            model(**enc)
    finally:
        for h in handles:
            h.remove()
    out = {k: {l: torch.cat(v).numpy() for l, v in d.items()} for k, d in store.items()}
    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez_compressed(path, **{f"{k}_{l}": out[k][l].astype(np.float16) for k in kinds for l in layer_ids})
        # 저장한 것과 같은 정밀도로 돌려준다(처음 실행과 다시 실행의 숫자가 같도록).
        out = {k: {l: out[k][l].astype(np.float16).astype(np.float32) for l in layer_ids} for k in kinds}
    return out


# ---------------------------------------------------------------------------
# 뉴런 끄기
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def set_neurons(model, layer: int, neurons: Sequence[int], values: Sequence[float] | float = 0.0) -> Iterator[None]:
    """층 `layer`의 뉴런 활성 a_i를 모든 위치에서 주어진 값으로 바꾼다(0 = zero ablation, 평균 = mean ablation)."""
    idx = torch.as_tensor(list(neurons), dtype=torch.long)
    vals = torch.as_tensor(values, dtype=torch.float32)

    def hook(m, inp):
        a = inp[0].clone()
        a[..., idx] = vals.to(a.dtype).to(a.device)
        return (a,)

    h = layers(model)[layer].mlp.down_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        h.remove()


# ---------------------------------------------------------------------------
# 읽기 도구
# ---------------------------------------------------------------------------
def neuron_aurocs(y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """뉴런마다 활성 하나를 점수로 쓴 AUROC. A: [N, F] → [F]. 0.5에서 먼 쪽이 잘 가른다(1 = 참에서 큼, 0 = 거짓에서 큼)."""
    from scipy.stats import rankdata

    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    r = rankdata(A, axis=0)  # 동점은 평균 순위
    return (r[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def random_unit_directions(dim: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cos_cols(v: np.ndarray, W: np.ndarray, axis: int) -> np.ndarray:
    """v와 W의 각 행(axis=1) 또는 각 열(axis=0) 사이의 cos."""
    v = v / np.linalg.norm(v)
    return (W @ v if axis == 1 else v @ W) / np.linalg.norm(W, axis=axis)


def logit_lens_of_vector(model, tok, vec: np.ndarray, k: int = 8) -> dict[str, list[str]]:
    """[근사] residual 방향 하나가 unembedding으로 어떤 token을 올리고 내리는지.
    마지막 RMSNorm의 크기 조정은 무시하고 weight(원소별 배율)만 곱한다."""
    W_U = model.lm_head.weight.detach().float().cpu().numpy()  # [V, D]
    g = model.model.norm.weight.detach().float().cpu().numpy()
    s = W_U @ (g * vec)
    top, bot = np.argsort(-s)[:k], np.argsort(s)[:k]
    dec = lambda ids: [tok.decode([int(i)]) for i in ids]
    return {"올림": dec(top), "내림": dec(bot)}


def save_results(name: str, results: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(pl._to_jsonable(results), ensure_ascii=False, indent=2))
    return path

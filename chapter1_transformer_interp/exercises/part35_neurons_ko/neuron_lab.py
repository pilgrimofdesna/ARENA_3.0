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


# ---------------------------------------------------------------------------
# ③ 희소화: 웹 문서, 보정 통계, 마스킹, 손실과 KL
# ---------------------------------------------------------------------------
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def wikitext_chunks(tok, split: str, n: int, T: int = 128, n_paragraphs: int = 3000) -> Tensor:
    """wikitext-103-raw-v1의 문단을 이어 붙여 T token씩 n개로 자른다. 제목 줄(= ... =)과 빈 줄은 뺀다.
    파일은 처음 한 번 Hugging Face에서 `cache/`로 받는다(CC BY-SA 3.0 / GFDL)."""
    import pandas as pd

    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"wikitext103_{split}.parquet"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        src = hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/{split}-00000-of-00001.parquet",
                              repo_type="dataset", revision=WIKITEXT_REVISION)
        path.write_bytes(Path(src).read_bytes())
    text = pd.read_parquet(path).text
    s = text.str.strip()
    text = text[(s.str.len() > 0) & (~s.str.startswith("="))]
    ids = tok(" ".join(text.tolist()[:n_paragraphs]), verbose=False)["input_ids"][: n * T]
    assert len(ids) == n * T, "문단이 모자란다: n_paragraphs를 늘린다"
    return torch.tensor(ids).view(n, T)


@torch.inference_mode()
def calibrate(model, chunks: Tensor, stride: int = 4, skip_first: bool = True, batch_size: int = 16):
    """층마다 뉴런 활성의 평균과 분위수(p3, p10, p90, p97)를 잰다.
    skip_first: 각 조각의 첫 위치를 뺀다. 첫 위치는 문맥이 없어 활성이 유난히 크거나 작다(③ Cell 03)."""
    L, F = model.config.num_hidden_layers, model.config.intermediate_size
    samp = {l: [] for l in range(L)}
    start = 1 if skip_first else 0
    hs = [layers(model)[l].mlp.down_proj.register_forward_pre_hook(
        lambda m, inp, l=l: samp[l].append(inp[0][:, start::stride].reshape(-1, F).clone())) for l in range(L)]
    try:
        for s in range(0, len(chunks), batch_size):
            model(chunks[s:s + batch_size])
    finally:
        for h in hs:
            h.remove()
    q = torch.tensor([0.03, 0.10, 0.90, 0.97])
    out = {}
    for l in range(L):
        X = torch.cat(samp[l])
        out[l] = dict(mean=X.mean(0), q=torch.quantile(X, q, dim=0))
    return out


QUANTILES = {"p3": 0, "p10": 1, "p90": 2, "p97": 3}


@contextlib.contextmanager
def sparsify(model, *, keep: str, k: int | None = None, band: tuple[str, str] | None = None,
             fill: str = "zero", stats: dict | None = None, seed: int = 0, record: list | None = None):
    """모든 층에서 token마다 일부 뉴런만 남기고 나머지를 0 또는 평균으로 바꾼다(down_proj 직전).

    keep="abs":    |a| 상위 k개를 남긴다 (논문의 top-K. 이 모델은 MoE가 아니라 router weight가 없다)
    keep="write":  |a|·‖d_i‖ 상위 k개 (쓰기 크기 기준. ① Cell 04의 크기 대칭 질문)
    keep="random": 무작위 k개 (대조군)
    keep="band":   뉴런마다 분위수 band=(lo, hi) 바깥(강한 쪽)만 남긴다. 예: ("p10", "p90")
    fill: "zero" 또는 "mean" (stats의 뉴런별 평균)
    record: 넘기면 남긴 비율을 batch마다 덧붙인다.
    """
    gen = torch.Generator().manual_seed(seed)
    dnorm = [layers(model)[l].mlp.down_proj.weight.norm(dim=0) for l in range(model.config.num_hidden_layers)]

    def make(l):
        def hook(m, inp):
            a = inp[0]
            if keep == "band":
                q = stats[l]["q"]
                mask = (a < q[QUANTILES[band[0]]]) | (a > q[QUANTILES[band[1]]])
            else:
                if keep == "abs":
                    score = a.abs()
                elif keep == "write":
                    score = a.abs() * dnorm[l]
                elif keep == "random":
                    score = torch.rand(a.shape, generator=gen)
                else:
                    raise ValueError(keep)
                idx = score.topk(k, dim=-1).indices
                mask = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, idx, True)
            filler = stats[l]["mean"].to(a.dtype) if fill == "mean" else torch.zeros((), dtype=a.dtype)
            if record is not None:
                record.append(mask.float().mean().item())
            return (torch.where(mask, a, filler),)
        return hook

    hs = [layers(model)[l].mlp.down_proj.register_forward_pre_hook(make(l)) for l in range(model.config.num_hidden_layers)]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@torch.inference_mode()
def loss_and_kl(model, chunks: Tensor, ctx=None, batch_size: int = 8) -> dict[str, float]:
    """다음 token 예측의 평균 cross-entropy(원래 모델, 바꾼 모델)와, 원래 분포에서 바꾼 분포로의 KL.
    ctx: 바꾼 모델을 만드는 context manager를 돌려주는 함수(예: lambda: sparsify(...)). None이면 원래 모델끼리."""
    nll_b, nll_a, kls = [], [], []
    for s in range(0, len(chunks), batch_size):
        x = chunks[s:s + batch_size]
        tgt = x[:, 1:, None]
        lpb = model(x).logits[:, :-1].float().log_softmax(-1)
        if ctx is None:
            lpa = lpb
        else:
            with ctx():
                lpa = model(x).logits[:, :-1].float().log_softmax(-1)
        nll_b.append(-lpb.gather(-1, tgt)[..., 0].flatten())
        nll_a.append(-lpa.gather(-1, tgt)[..., 0].flatten())
        kls.append((lpb.exp() * (lpb - lpa)).sum(-1).flatten())
        del lpb, lpa
    b, a = torch.cat(nll_b).mean().item(), torch.cat(nll_a).mean().item()
    return dict(loss_base=b, loss=a, rel=a / b - 1, kl=torch.cat(kls).mean().item())

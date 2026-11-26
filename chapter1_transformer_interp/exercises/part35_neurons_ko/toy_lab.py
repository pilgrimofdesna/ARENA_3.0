"""[뉴런을 읽기 ②] 장난감 superposition 모델 (CPU용).

Elhage et al. 2022, *Toy Models of Superposition*의 설정을 ARENA 1.5.4 §1·§2·§5의 순서로 다시 짰다. 코드는 새로 썼다.

    특징 x ∈ [0,1]^n  (각 특징은 확률 1−S로 켜지고, 켜지면 균등분포 값)
    은닉 h = W x       (비특권 기저) 또는 h = ReLU(W x)  (특권 기저: 은닉 좌표마다 ReLU)
    복원 x' = ReLU(Wᵀ h + b)
    손실 = Σ_i I_i (x_i − x'_i)²  (I_i = 중요도)

여러 설정(instance)을 한 번에 학습한다. W: [instance, m, n].
"""
from __future__ import annotations

import torch

torch.set_num_threads(4)


def indep_data(n_inst: int, n_feat: int, S):
    """특징이 서로 독립. S: instance마다의 희소도(꺼져 있을 확률)."""
    S = torch.as_tensor(S, dtype=torch.float32)

    def f(bs, g):
        x = torch.rand(bs, n_inst, n_feat, generator=g)
        return x * (torch.rand(bs, n_inst, n_feat, generator=g) > S[None, :, None])

    return f


def group_data(n_inst: int, n_groups: int, size: int, p_on: float):
    """특징이 묶음으로 함께 켜진다. 묶음 하나가 확률 p_on으로 켜지면 그 안의 특징 `size`개가 모두 켜지고 값은 각자 균등분포."""

    def f(bs, g):
        on = (torch.rand(bs, n_inst, n_groups, generator=g) < p_on).float()
        x = torch.rand(bs, n_inst, n_groups, size, generator=g) * on[..., None]
        return x.reshape(bs, n_inst, n_groups * size)

    return f


def forward(W, b, x, priv: bool):
    h = torch.einsum("imf,bif->bim", W, x)
    if priv:
        h = torch.relu(h)
    return h, torch.relu(torch.einsum("imf,bim->bif", W, h) + b)


def train(n_inst, n_feat, m, data, priv=False, imp=None, steps=8000, bs=1024, lr=1e-3, seed=0):
    """Returns W [inst, m, n], b [inst, n], 손실 [inst] (새 batch 8,192개로 잰 값)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(n_inst, m, n_feat) * (2 / (m + n_feat)) ** 0.5)
    b = torch.nn.Parameter(torch.zeros(n_inst, n_feat))
    I = torch.ones(n_feat) if imp is None else imp ** torch.arange(n_feat).float()
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        x = data(bs, g)
        _, out = forward(W, b, x, priv)
        loss = (I * (out - x) ** 2).mean((0, 2))
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        x = data(8192, g)
        _, out = forward(W, b, x, priv)
        loss = (I * (out - x) ** 2).mean((0, 2))
    return W.detach(), b.detach(), loss


def best_of(seeds, **kw):
    """seed 여러 개로 학습하고 instance마다 손실이 가장 낮은 것을 고른다(나쁜 국소 최솟값을 피하려고)."""
    best = None
    for s in seeds:
        W, b, L = train(seed=s, **kw)
        if best is None:
            best = [W.clone(), b.clone(), L.clone()]
        else:
            m = L < best[2]
            best[0][m], best[1][m], best[2][m] = W[m], b[m], L[m]
    return best


def interference(W):
    """W [m, n]. 특징 i가 다른 특징과 겹치는 정도 Σ_{j≠i} (Ŵ_i · W_j)²."""
    n = W.norm(dim=0)
    G = ((W / n.clamp(min=1e-8)).T @ W) ** 2
    return G.sum(1) - torch.diagonal(G)


def train_sae(W, S, n_lat=20, l1=0.2, steps=10000, lr=3e-3, seed=0):
    """특권 기저 모델의 은닉 h = ReLU(W x)에 SAE를 학습한다. 디코더 방향은 단위 길이로 묶는다.
    Returns dict(W_dec [n_lat, m] 단위 벡터, fvu, alive, L0)."""
    m, n = W.shape
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    data = indep_data(1, n, [S])
    We = torch.nn.Parameter(torch.randn(m, n_lat) * 0.3)
    be = torch.nn.Parameter(torch.zeros(n_lat))
    Wd = torch.nn.Parameter(torch.randn(n_lat, m) * 0.3)
    bd = torch.nn.Parameter(torch.zeros(m))
    opt = torch.optim.Adam([We, be, Wd, bd], lr=lr)
    for _ in range(steps):
        h = torch.relu(data(1024, g)[:, 0] @ W.T)
        Wdn = Wd / Wd.norm(dim=1, keepdim=True)
        z = torch.relu((h - bd) @ We + be)
        loss = ((z @ Wdn + bd - h) ** 2).sum(1).mean() + l1 * z.sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        Wdn = Wd / Wd.norm(dim=1, keepdim=True)
        h = torch.relu(data(8192, g)[:, 0] @ W.T)
        z = torch.relu((h - bd) @ We + be)
        fvu = ((z @ Wdn + bd - h) ** 2).sum() / ((h - h.mean(0)) ** 2).sum()
    return dict(W_dec=Wdn, fvu=float(fvu), alive=int((z > 0).any(0).sum()), L0=float((z > 0).float().sum(1).mean()))

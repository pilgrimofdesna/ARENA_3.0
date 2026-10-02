"""[1.3.1 한국어판] exercise 확인용 테스트.

각 notebook의 scratch 셀에서 직접 구현한 함수를 넘기면 reference(`probe_lab`)와 비교한다.
통과하면 "✅ ..."를 출력하고, 어긋나면 AssertionError에 이유를 적는다.
"""

from __future__ import annotations

import numpy as np
import torch

import probe_lab as pl


def _toy(n: int = 400, d: int = 20, seed: int = 0, signal: float = 1.5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d))
    X[:, 0] += signal * (2 * y - 1)
    X[:, 1] += 3.0 * X[:, 0] + rng.normal(size=n)  # 상관된 nuisance 축
    X += 7.0  # 원점에서 멀리 떨어진 데이터
    return X, y


def test_fit_mass_mean(fn) -> None:
    X, y = _toy()
    ref = pl.fit_mass_mean(X, y)
    got = fn(X, y)
    assert np.allclose(got.direction, ref.direction, atol=1e-6), "direction은 μ₁ − μ₀이어야 한다."
    assert abs(got.bias - ref.bias) < 1e-6, "bias는 −θ·(μ₁+μ₀)/2 (두 평균의 중점)이어야 한다."
    shift = np.full(X.shape[1], 13.0)
    moved = fn(X + shift, y)
    assert np.allclose(moved.score(X + shift), got.score(X), atol=1e-6), (
        "데이터 전체를 평행 이동해도 판정이 같아야 한다(중점 bias가 빠지면 깨진다)."
    )
    print("✅ fit_mass_mean: 방향·중점 bias·평행 이동 불변성 통과")


def test_fit_logistic(fn) -> None:
    X, y = _toy()
    ref = pl.fit_logistic(X, y, C=0.1)
    got = fn(X, y, C=0.1)
    cos = pl.cosine(got.direction, ref.direction)
    assert cos > 0.99, f"raw 좌표 방향이 reference와 다르다 (cos={cos:.3f}). w_raw = w_z / s 를 확인하라."
    r = np.corrcoef(got.score(X), ref.score(X))[0, 1]
    assert r > 0.999, "score가 reference와 다르다. b_raw = b_z − w_raw·m 을 확인하라."
    print(f"✅ fit_logistic: raw 좌표 방향 cos={cos:.4f}, score 상관 {r:.4f}")


def test_extract_activations(fn, model, tokenizer) -> None:
    texts = [
        "The city of Paris is in France.",
        "The city of Lagos is in Nigeria.",
        "Tokyo.",
        "The city of San Francisco is in the United States of America.",
    ]
    layers = [0, 12]
    ref = pl.extract_activations(model, tokenizer, texts, layers, batch_size=4)
    got = fn(model, tokenizer, texts, layers, batch_size=4)
    for l in layers:
        g = torch.as_tensor(np.asarray(got[l]), dtype=torch.float32)
        assert g.shape == ref[l].shape, f"layer {l}: shape {tuple(g.shape)} ≠ {tuple(ref[l].shape)}"
        err = (g - ref[l]).abs().max().item()
        assert err < 1e-2, (
            f"layer {l}: 최대 오차 {err:.3g}. 길이가 다른 문장에서 padding 위치를 읽고 있지 않은지, "
            "hidden_states 대신 block 출력을 읽는지 확인하라."
        )
    print("✅ extract_activations: 길이가 다른 문장 4개, layer 0·12에서 reference와 일치")


def test_permutation_null(fn) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 50))
    y = rng.integers(0, 2, 300)
    Xte = rng.normal(size=(200, 50))
    yte = rng.integers(0, 2, 200)
    yte_before = yte.copy()
    null = np.asarray(fn(pl.fit_mass_mean, X, y, Xte, yte, n_perm=50, seed=0))
    assert null.shape == (50,), "n_perm개의 AUROC를 돌려줘야 한다."
    assert np.array_equal(yte, yte_before), "test label을 바꾸면 안 된다."
    assert 0.4 < null.mean() < 0.6, "신호가 없는 데이터에서 귀무분포 평균은 0.5 근처여야 한다."
    print(f"✅ permutation_null: shape OK, test label 보존, 무신호 평균 {null.mean():.3f}")


def test_true_false_margin(fn, model, tokenizer) -> None:
    prompts = pl.build_prompts(["The city of Paris is in France.", "The city of Paris is in Japan."])
    ref = pl.true_false_margin(model, tokenizer, prompts)
    got = fn(model, tokenizer, prompts)
    assert np.allclose(np.asarray(got), ref["p_diff"], atol=2e-2), (
        f"P(TRUE)−P(FALSE)가 다르다: {np.asarray(got)} vs {ref['p_diff']}. "
        "padding 때문에 마지막 위치가 어긋나지 않았는지, ' TRUE'(앞 공백 포함) token id를 썼는지 확인하라."
    )
    print(f"✅ true_false_margin: {np.round(ref['p_diff'], 3)}")


def test_fit_truth_polarity(fn) -> None:
    rng = np.random.default_rng(0)
    d = 30
    tG = rng.normal(size=d)
    tP = rng.normal(size=d)
    Xs, ys = {}, {}
    for name, pol in [("a", 1), ("neg_a", -1), ("b", 1), ("neg_b", -1)]:
        y = rng.integers(0, 2, 300)
        tau = 2 * y - 1
        X = rng.normal(scale=0.3, size=(300, d)) + tau[:, None] * tG + (tau * pol)[:, None] * tP + rng.normal(size=d)
        Xs[name], ys[name] = X, y
    g, p = fn(Xs, ys)
    assert pl.cosine(g, tG) > 0.99 and pl.cosine(p, tP) > 0.99, "합성 데이터에서 t_G, t_P를 복원하지 못했다."
    print("✅ fit_truth_polarity: 합성 데이터의 t_G, t_P 복원 (cos > 0.99)")


def test_eraser(fn) -> None:
    X, y = _toy(n=600, d=20)
    erase = fn(X, y)
    Xe = erase(X)
    probe = pl.fit_logistic(Xe, y, C=1.0)
    au = pl.auroc(y, probe.score(Xe))
    assert au < 0.6, f"지운 뒤에도 같은 데이터에서 선형 probe가 label을 읽는다 (AUROC {au:.3f})."
    change = np.linalg.norm(Xe - X, axis=1).mean() / np.linalg.norm(X - X.mean(0), axis=1).mean()
    print(f"✅ eraser: 지운 뒤 재학습 LR AUROC {au:.3f}, 평균 변화량 {change:.2f} (중심화된 norm 대비)")


def test_tpr_at_fpr(fn) -> None:
    rng = np.random.default_rng(0)
    neg = rng.normal(size=1000)
    pos = rng.normal(loc=2.0, size=500)
    y = np.r_[np.zeros(1000), np.ones(500)].astype(int)
    s = np.r_[neg, pos]
    ref_tpr, ref_thr = pl.tpr_at_fpr(y, s, 0.01)
    got_tpr, got_thr = fn(y, s, 0.01)
    assert abs(got_thr - ref_thr) < 0.05 and abs(got_tpr - ref_tpr) < 0.02, (
        "threshold는 음성 score의 (1 − fpr) 분위수여야 한다."
    )
    print(f"✅ tpr_at_fpr: threshold {got_thr:.3f}, TPR {got_tpr:.3f}")

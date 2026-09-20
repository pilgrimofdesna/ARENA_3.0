"""Small, reproducible Llama-2-13B experiment; no secrets, no automatic downloads.

Run using /venv/main/bin/python run_truth_pilot.py. Published outputs are a pilot,
not a reproduction of the paper's full experiments.
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import subprocess
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from probe_lab import (group_split_indices, fit_mean_difference, fit_logistic_probe,
                       score_metrics, extract_last_token_states, group_bootstrap_ci,
                       next_token_contrast)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data/geometry-of-truth"
OUT = HERE / "artifacts"
SEED = 42
LAYERS = [8, 14, 20, 28, 36]  # Block outputs, zero indexed, before final norm.
MODEL = "meta-llama/Llama-2-13b-hf"


def prepare_data():
    rng = np.random.default_rng(SEED)
    cities = pd.read_csv(DATA / "cities.csv")
    selected = rng.choice(cities.city.unique(), 120, replace=False)
    cities = cities[cities.city.isin(selected)].reset_index(drop=True)
    split = group_split_indices(cities.city.to_numpy(), seed=SEED)
    cities["split"] = ""
    for name in ("train", "val", "test"):
        cities.loc[getattr(split, name), "split"] = name
    cities["domain"] = "cities"
    cities["group"] = cities.city
    neg = pd.read_csv(DATA / "neg_cities.csv")
    # Only unseen city entities enter the negation stress test.
    test_cities = cities.loc[cities.split == "test", "city"]
    neg = neg[neg.city.isin(test_cities)].copy()
    neg["domain"], neg["split"], neg["group"] = "neg_cities", "ood", neg.city
    sp = pd.read_csv(DATA / "sp_en_trans.csv")
    assert sp.groupby("statement").label.nunique().max() == 1, "Conflicting labels require manual review."
    sp = sp.drop_duplicates("statement")  # Upstream repeats three positive translations.
    sp["group"] = sp.statement.str.extract(r"word '([^']+)'", expand=False)
    words = rng.choice(sp.group.unique(), min(60, sp.group.nunique()), replace=False)
    sp = sp[sp.group.isin(words)].copy()
    sp["domain"], sp["split"] = "sp_en_trans", "ood"
    num = pd.read_csv(DATA / "larger_than.csv")
    num["group"] = [f"{min(a,b)}:{max(a,b)}" for a,b in zip(num.n1,num.n2)]
    pairs = rng.choice(num.group.unique(), 60, replace=False)
    num = num[num.group.isin(pairs)].copy()
    num["domain"], num["split"] = "larger_than", "ood"
    result = pd.concat([cities, neg, sp, num], ignore_index=True)
    assert result.statement.notna().all() and result.group.notna().all()
    assert not result.statement.duplicated().any()
    for name in ["train", "val", "test"]:
        assert set(result.loc[result.split == name, "label"]) == {0,1}
    return result[["statement", "label", "group", "domain", "split"]]


def main():
    start = time.time()
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    df = prepare_data()
    df.to_csv(OUT / "data_manifest.csv", index=False)
    paths = json.loads((HERE / "model_paths.local.json").read_text())
    local = paths[MODEL]
    print("Load", MODEL, "from verified local cache", flush=True)
    tok = AutoTokenizer.from_pretrained(local, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        local, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa", use_safetensors=True,
    ).eval()
    model.requires_grad_(False)
    cache = {layer: [] for layer in LAYERS}
    token_lengths = []
    for lo in range(0, len(df), 8):
        inputs = tok(df.statement.iloc[lo:lo+8].tolist(), padding=True,
                     return_tensors="pt", truncation=False).to("cuda")
        assert inputs.input_ids.shape[1] <= 128, "Inspect long inputs; no silent truncation."
        token_lengths.extend(inputs.attention_mask.sum(1).cpu().tolist())
        states = extract_last_token_states(model, inputs.input_ids, inputs.attention_mask, LAYERS)
        for layer in LAYERS:
            cache[layer].append(states[layer].numpy())
        if lo % 80 == 0:
            print(f"Extracted {min(lo+8,len(df))}/{len(df)}", flush=True)
    cache = {k: np.concatenate(v) for k,v in cache.items()}
    np.savez_compressed(OUT / "activations.npz", **{f"layer_{k}":v for k,v in cache.items()})
    y = df.label.to_numpy()
    train, val = np.flatnonzero(df.split == "train"), np.flatnonzero(df.split == "val")
    selection, fitted = [], {}
    # Nothing in this selection loop reads test or OOD labels.
    for layer, X in cache.items():
        for kind, C in [("MM", None)] + [("LR", c) for c in [0.01, 0.1, 1.0]]:
            p = fit_mean_difference(X[train], y[train]) if kind == "MM" else fit_logistic_probe(X[train], y[train], C=C, seed=SEED)
            key = (layer, kind, C)
            fitted[key] = p
            m = score_metrics(y[val], p.decision_function(X[val]))
            selection.append(dict(layer=layer, probe=kind, C=C, **m))
    selection_df = pd.DataFrame(selection)
    selection_df.to_csv(OUT / "validation_selection.csv", index=False)
    # Deterministic tie-break preserves declared layer/C order.
    best_rows = {kind: max([r for r in selection if r["probe"] == kind], key=lambda r:r["auroc"]) for kind in ["MM", "LR"]}
    results = []
    for kind, row in best_rows.items():
        layer, C = row["layer"], row["C"]
        p = fitted[(layer, kind, C)]
        np.savez(OUT / f"selected_{kind.lower()}.npz", direction=p.direction, bias=p.bias, layer=layer)
        for domain in df.domain.unique():
            idx = np.flatnonzero((df.domain == domain) & df.split.isin(["test", "ood"]))
            scores = np.asarray(p.decision_function(cache[layer][idx]))
            metrics = score_metrics(y[idx], scores)
            ci = group_bootstrap_ci(y[idx], scores, df.group.to_numpy()[idx], n_resamples=500, seed=SEED)
            results.append(dict(probe=kind, layer=layer, domain=domain, n=len(idx),
                                groups=int(df.iloc[idx].group.nunique()), **metrics,
                                auroc_ci=list(ci)))
    # Predeclared nuisance baseline: token count only. No label-derived numeric diff.
    lengths = np.array(token_lengths).reshape(-1,1)
    length_probe = fit_logistic_probe(lengths[train], y[train], C=1.0, seed=SEED)
    baseline = []
    for domain in df.domain.unique():
        idx = np.flatnonzero((df.domain==domain) & df.split.isin(["test","ood"]))
        baseline.append(dict(domain=domain, **score_metrics(y[idx],length_probe.decision_function(lengths[idx]))))
    # Null control at one predeclared layer, not a best-of-many permutation search.
    rng = np.random.default_rng(SEED)
    nulls = []
    test = np.flatnonzero(df.split == "test")
    for _ in range(20):
        p = fit_mean_difference(cache[14][train], rng.permutation(y[train]))
        nulls.append(score_metrics(y[test],p.decision_function(cache[14][test]))["auroc"])

    print("Starting held-out additive intervention sanity check", flush=True)
    layer = 14
    p = fitted[(layer, "MM", None)]
    direction = np.asarray(p.direction)
    direction = direction / np.linalg.norm(direction)
    # Fixed scale = one training projection SD, no test-optimised steering strength.
    scale = float(np.std(cache[layer][train] @ direction))
    # 12 paired rows, deterministic. This is a tiny sensitivity check, not NIE.
    examples = df.iloc[test[:12]].copy()
    prefix = "Decide if each statement is true or false.\nStatement: The city of Paris is in France.\nAnswer: True\nStatement: The city of Tokyo is in Brazil.\nAnswer: False\n"
    prompts = [prefix + "Statement: " + s + "\nAnswer:" for s in examples.statement]
    inp = tok(prompts, padding=True, return_tensors="pt").to("cuda")
    # SentencePiece's isolated leading space can insert an extra whitespace token.
    # Resolve continuation IDs in the full prompt context instead.
    prompt_ids = tok.encode(prompts[0], add_special_tokens=True)
    pos_full = tok.encode(prompts[0] + " True", add_special_tokens=True)
    neg_full = tok.encode(prompts[0] + " False", add_special_tokens=True)
    assert pos_full[:len(prompt_ids)] == neg_full[:len(prompt_ids)] == prompt_ids
    pos, neg = pos_full[len(prompt_ids):], neg_full[len(prompt_ids):]
    assert len(pos) == len(neg) == 1, "Use full continuation log-prob if labels split into multiple tokens."
    for prompt in prompts:
        ids = tok.encode(prompt, add_special_tokens=True)
        assert tok.encode(prompt+" True",add_special_tokens=True) == ids+pos
        assert tok.encode(prompt+" False",add_special_tokens=True) == ids+neg
    random_direction = rng.normal(size=direction.shape)
    random_direction /= np.linalg.norm(random_direction)
    base = next_token_contrast(model, inp.input_ids, inp.attention_mask, pos[0], neg[0]).numpy()
    effects = []
    for name, vec in [("MM", direction), ("random", random_direction)]:
        for alpha in [-2., -1., 0., 1., 2.]:
            scores = next_token_contrast(model, inp.input_ids, inp.attention_mask,
                        pos[0], neg[0], layer_index=layer, direction=vec, strength=alpha*scale).numpy()
            if alpha == 0:
                assert np.allclose(scores,base,atol=.1), "Zero intervention changed baseline unexpectedly."
            effects.append(dict(direction=name, alpha=alpha, mean_logit_shift=float((scores-base).mean()),
                                mean_true_logit_margin=float(scores.mean()), n=len(base)))
    revisions = json.loads((HERE / "source_revisions.json").read_text())
    metadata = dict(source_commit="4605b1fb676dcf2ba0704c7821b19a0a58f4484a",
        data_commit=revisions["geometry-of-truth"]["commit"],
        model=MODEL, model_revision=Path(local).name, seed=SEED, layers=LAYERS,
        selection="validation AUROC, deterministic ties; train-only scaling/PCA/probes",
        split_counts=df.groupby(["domain","split"]).size().to_dict().__repr__(),
        dataset_sha256=hashlib.sha256((OUT/"data_manifest.csv").read_bytes()).hexdigest(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0), dtype="bfloat16", batch_size=8,
        max_gpu_GiB=round(torch.cuda.max_memory_allocated()/2**30,2), elapsed_seconds=round(time.time()-start),
        selected=best_rows, results=results, token_length_baseline=baseline,
        shuffled_label_auroc=nulls, intervention=effects,
        intervention_scale=scale, intervention_location="block 14, final Answer: prompt token",
        limitations=["Small pilot, single seed/model, wide group bootstrap intervals.",
          "Truth direction trained on statement punctuation; intervention uses different few-shot Answer: context.",
          "One random direction is a sanity check, not a null distribution.",
          "Additive sensitivity is neither activation patching nor a natural indirect effect.",
          "Negation stress test reuses held-out city facts, so domains are statistically dependent."])
    (OUT / "results.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    pd.DataFrame(results).to_csv(OUT/"test_metrics.csv",index=False)
    pd.DataFrame(effects).to_csv(OUT/"intervention.csv",index=False)
    print(json.dumps({"elapsed_seconds":metadata["elapsed_seconds"],"results":results},indent=2),flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Small, auditable helpers for the Korean linear-probe lab.

All fitting functions receive *training data only*. Callers select hyperparameters
on validation data and reserve the test set for the final comparison. Transformer
layer indices refer to zero-based decoder blocks, not ``hidden_states`` indices.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


@dataclass(frozen=True)
class GroupSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def group_split_indices(groups, train_fraction=0.6, val_fraction=0.2, seed=0):
    """Split whole groups, never rows; fractions describe numbers of groups.

    At least three groups and positive train/validation/test fractions are needed.
    Class balance is deliberately not guaranteed: inspect every resulting split.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be a one-dimensional array")
    if np.issubdtype(groups.dtype, np.number) and not np.isfinite(groups).all():
        raise ValueError("group identifiers cannot contain NaN or infinity")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1
            and train_fraction + val_fraction < 1):
        raise ValueError("train/val/test fractions must all be positive")
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("at least three distinct groups are required")
    unique = np.random.default_rng(seed).permutation(unique)
    n_train = min(max(1, int(len(unique) * train_fraction)), len(unique) - 2)
    n_val = min(max(1, int(len(unique) * val_fraction)), len(unique) - n_train - 1)
    return GroupSplit(
        train=np.flatnonzero(np.isin(groups, unique[:n_train])),
        val=np.flatnonzero(np.isin(groups, unique[n_train:n_train + n_val])),
        test=np.flatnonzero(np.isin(groups, unique[n_train + n_val:])),
    )


def _features(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite [examples, features] array")
    return X


def _labels(y, n, require_both=False):
    y = np.asarray(y)
    if y.shape != (n,) or not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain one binary 0/1 label per example")
    if require_both and len(np.unique(y)) != 2:
        raise ValueError("training labels must contain both classes")
    return y.astype(np.int64)


@dataclass(frozen=True)
class LinearProbe:
    """A linear classifier in the original, unstandardized activation space.

    ``predict_proba`` returns [N, 2]. Its sigmoid values are calibrated logistic
    probabilities only to the extent supported by held-out calibration evidence;
    for a mean-difference probe they are merely a monotone score transform.
    """
    direction: np.ndarray
    bias: float

    def decision_function(self, X):
        X = _features(X)
        if X.shape[1] != self.direction.shape[0]:
            raise ValueError("feature dimension does not match this probe")
        return X @ self.direction + self.bias

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.int64)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        positive = np.exp(-np.logaddexp(0, -scores))
        return np.column_stack((1 - positive, positive))


def fit_mean_difference(X, y):
    """Use mean(class 1) - mean(class 0) and the projected midpoint threshold."""
    X = _features(X)
    y = _labels(y, len(X), require_both=True)
    negative_mean, positive_mean = X[y == 0].mean(0), X[y == 1].mean(0)
    direction = positive_mean - negative_mean
    bias = -float(((positive_mean + negative_mean) / 2) @ direction)
    return LinearProbe(direction=direction, bias=bias)


def fit_logistic_probe(X, y, C=1, seed=0):
    """Fit train-only standardization and logistic regression; map back to raw X.

    Standardized weights cannot be used directly as an activation intervention.
    ``direction = coef / scale`` and ``bias = intercept - direction @ mean``
    preserve the fitted classifier's decisions in the original coordinates.
    """
    X = _features(X)
    y = _labels(y, len(X), require_both=True)
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be positive and finite")
    scaler = StandardScaler().fit(X)
    classifier = LogisticRegression(C=C, random_state=seed, max_iter=3000)
    classifier.fit(scaler.transform(X), y)
    direction = classifier.coef_[0] / scaler.scale_
    bias = float(classifier.intercept_[0] - direction @ scaler.mean_)
    return LinearProbe(direction=direction, bias=bias)


def score_metrics(y, scores, threshold=0):
    """Evaluate fixed scores; choose threshold on validation data, never test data.

    AUROC and two-class balanced accuracy are NaN when only one class is present.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite, nonempty one-dimensional array")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    y = _labels(y, len(scores))
    predictions = scores >= threshold
    both_classes = len(np.unique(y)) == 2
    return {
        "auroc": float(roc_auc_score(y, scores)) if both_classes else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)) if both_classes else float("nan"),
        "accuracy": float(accuracy_score(y, predictions)),
    }


@dataclass(frozen=True)
class TrainPCA:
    mean: np.ndarray
    components: np.ndarray

    def transform(self, X):
        X = _features(X)
        if X.shape[1] != self.mean.shape[0]:
            raise ValueError("feature dimension does not match this PCA")
        return (X - self.mean) @ self.components.T


def fit_train_pca(X, n_components=2):
    """Fit centered thin SVD using training activations only."""
    X = _features(X)
    if (not isinstance(n_components, (int, np.integer))
            or not 1 <= n_components <= min(X.shape)):
        raise ValueError("n_components must lie between 1 and min(X.shape)")
    mean = X.mean(0)
    _, _, components = np.linalg.svd(X - mean, full_matrices=False)
    return TrainPCA(mean=mean, components=components[:n_components].copy())


def group_bootstrap_ci(y, scores, groups, metric="auroc", n_resamples=1000, seed=0):
    """Return (low, high), a 95% percentile CI from resampling whole groups.

    This conditions on the fitted model/probe; it does not quantify variability
    from refitting or model selection. Rows within a sampled group stay together.
    The metrics use threshold 0; AUROC/balanced-accuracy resamples containing only
    one class are skipped because the two-class metric is undefined there.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite, nonempty and one-dimensional")
    y = _labels(y, len(scores))
    groups = np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError("one group identifier per example is required")
    if np.issubdtype(groups.dtype, np.number) and not np.isfinite(groups).all():
        raise ValueError("group identifiers cannot contain NaN or infinity")
    if metric not in {"auroc", "balanced_accuracy", "accuracy"}:
        raise ValueError("metric must be auroc, balanced_accuracy, or accuracy")
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("a group bootstrap requires at least two groups")
    if metric != "accuracy" and len(np.unique(y)) != 2:
        raise ValueError("AUROC and balanced accuracy require both classes")
    rows = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        indices = np.concatenate([rows[i] for i in rng.integers(len(rows), size=len(rows))])
        if metric != "accuracy" and len(np.unique(y[indices])) != 2:
            continue
        if metric == "auroc":
            value = roc_auc_score(y[indices], scores[indices])
        elif metric == "accuracy":
            value = accuracy_score(y[indices], scores[indices] >= 0)
        else:
            value = balanced_accuracy_score(y[indices], scores[indices] >= 0)
        values.append(value)
    if not values:
        raise ValueError("no valid bootstrap resamples; increase n_resamples or inspect groups")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def last_valid_token_indices(mask):
    """Return [batch] positions, supporting both left and right padding."""
    mask = torch.as_tensor(mask)
    if mask.ndim != 2 or 0 in mask.shape:
        raise ValueError("attention mask must have nonempty shape [batch, sequence]")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("attention mask must contain only 0/1")
    valid = mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every sequence must contain at least one valid token")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return positions.masked_fill(~valid, -1).max(dim=1).values


def _decoder_layers(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("expected a causal LM exposing model.model.layers")
    return model.model.layers


def _check_layer(layer_index, layers):
    if not isinstance(layer_index, (int, np.integer)) or not 0 <= layer_index < len(layers):
        raise ValueError(f"layer_index must be a zero-based block index in [0, {len(layers)})")
    return int(layer_index)


def _block_hidden(output):
    hidden = output if isinstance(output, torch.Tensor) else output[0]
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("decoder block must output hidden states [batch, sequence, d_model]")
    return hidden


def _model_inputs(model, input_ids, attention_mask):
    input_ids, attention_mask = torch.as_tensor(input_ids), torch.as_tensor(attention_mask)
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    positions = last_valid_token_indices(attention_mask)
    try:
        device = model.get_input_embeddings().weight.device
    except (AttributeError, NotImplementedError):
        device = next(model.parameters()).device
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}, positions


@contextmanager
def _evaluation(model):
    # Preserve mixed train/eval submodule settings as well as the top-level flag.
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in states:
            module.training = training


def extract_last_token_states(model, input_ids, attention_mask, layer_indices):
    """Capture post-block residual states as CPU float32 [batch, d_model].

    Runs ``model.model`` rather than the LM head to avoid allocating vocabulary
    logits. Hooks are removed even when model execution fails. No network is used.
    """
    layers = _decoder_layers(model)
    indices = list(dict.fromkeys(_check_layer(i, layers) for i in layer_indices))
    inputs, positions = _model_inputs(model, input_ids, attention_mask)
    if not indices:
        return {}
    result, handles = {}, []

    def capture(index):
        def hook(_module, _args, output):
            hidden = _block_hidden(output)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            result[index] = hidden[batch, positions.to(hidden.device)].detach().to("cpu", torch.float32, copy=True)
        return hook

    try:
        for index in indices:
            handles.append(layers[index].register_forward_hook(capture(index)))
        with _evaluation(model):
            model.model(**inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return result


class AttentionProbe(nn.Module):
    """A learned-query pooling classifier; unlike mean/LR probes, it is nonlinear.

    Use a held-out validation set for early stopping. Compare against equal-budget
    mean pooling and last-token baselines before attributing gains to attention.
    """
    def __init__(self, d_model):
        super().__init__()
        if not isinstance(d_model, int) or d_model < 1:
            raise ValueError("d_model must be a positive integer")
        self.query = nn.Parameter(torch.zeros(d_model))
        self.classifier = nn.Linear(d_model, 1)

    def get_attention_weights(self, hidden, mask):
        """Return [batch, sequence] pooling weights, with zero weight on padding."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.query.numel():
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        mask = torch.as_tensor(mask, device=hidden.device)
        if mask.shape != hidden.shape[:2]:
            raise ValueError("mask must match the first two hidden dimensions")
        last_valid_token_indices(mask)  # Reject fully padded rows and nonbinary masks.
        scores = hidden @ self.query / math.sqrt(self.query.numel())
        return scores.masked_fill(~mask.bool(), -torch.inf).softmax(dim=1)

    def forward(self, hidden, mask):
        weights = self.get_attention_weights(hidden, mask)
        pooled = (weights.unsqueeze(-1) * hidden).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


def next_token_contrast(model, input_ids, attention_mask, positive_token_id,
                        negative_token_id, layer_index=None, direction=None, strength=0):
    """Return the positive-minus-negative next-token logit contrast, CPU float32.

    Optionally add ``strength * direction / ||direction||`` at the last valid
    token of one decoder block. Direction is in raw activation coordinates.
    A changed contrast is a local causal effect, not proof of a truth mechanism.
    """
    if not np.isfinite(strength):
        raise ValueError("strength must be finite")
    if not all(isinstance(i, (int, np.integer)) and i >= 0
               for i in [positive_token_id, negative_token_id]):
        raise ValueError("token IDs must be nonnegative integers")
    inputs, positions = _model_inputs(model, input_ids, attention_mask)
    handle = None
    if strength != 0:
        if layer_index is None or direction is None:
            raise ValueError("a nonzero intervention needs layer_index and direction")
        layers = _decoder_layers(model)
        layer_index = _check_layer(layer_index, layers)
        vector = torch.as_tensor(direction, dtype=torch.float32)
        if vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm() == 0:
            raise ValueError("direction must be a finite, nonzero one-dimensional vector")
        vector = vector / vector.norm()

        def intervene(_module, _args, output):
            hidden = _block_hidden(output)
            if hidden.shape[-1] != vector.numel():
                raise ValueError("direction dimension must match hidden states")
            changed = hidden.clone()
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            changed[batch, positions.to(hidden.device)] += strength * vector.to(hidden)
            if isinstance(output, torch.Tensor):
                return changed
            if isinstance(output, tuple):
                return (changed, *output[1:])
            return [changed, *output[1:]]

        handle = layers[layer_index].register_forward_hook(intervene)
    try:
        with _evaluation(model):
            output = model(**inputs, use_cache=False, return_dict=True)
            logits = output.logits
            if max(positive_token_id, negative_token_id) >= logits.shape[-1]:
                raise ValueError("token ID is outside this model's vocabulary")
            batch = torch.arange(logits.shape[0], device=logits.device)
            final_logits = logits[batch, positions.to(logits.device)].float()
            contrast = final_logits[:, positive_token_id] - final_logits[:, negative_token_id]
            return contrast.detach().to("cpu", torch.float32)
    finally:
        if handle is not None:
            handle.remove()

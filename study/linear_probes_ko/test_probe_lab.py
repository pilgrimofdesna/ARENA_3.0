"""CPU-only regression checks; no model downloads or credentials are needed.

Run: python -m unittest discover -s study/linear_probes_ko -p test_probe_lab.py -v
"""

from types import SimpleNamespace
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from probe_lab import (
    AttentionProbe,
    extract_last_token_states,
    fit_logistic_probe,
    fit_mean_difference,
    fit_train_pca,
    group_bootstrap_ci,
    group_split_indices,
    last_valid_token_indices,
    next_token_contrast,
    score_metrics,
)


class StatisticalHelpersTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = rng.normal(size=(120, 4)) * [0.1, 7, 1, 0]
        self.X += [30, -200, 1, 9]
        self.y = (self.X[:, 0] - 30 + self.X[:, 1] / 70 + 2.8 > 0).astype(int)

    def test_group_split_keeps_templates_and_paraphrases_together(self):
        groups = np.repeat(np.arange(15), np.arange(1, 16))
        split = group_split_indices(groups, seed=31)
        split_again = group_split_indices(groups, seed=31)
        parts = [split.train, split.val, split.test]
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(groups)))
        for first, second in [(0, 1), (0, 2), (1, 2)]:
            self.assertTrue(set(groups[parts[first]]).isdisjoint(groups[parts[second]]))
        np.testing.assert_array_equal(split.train, split_again.train)
        smallest = group_split_indices(["a", "a", "b", "c"])
        self.assertTrue(all(len(part) for part in [smallest.train, smallest.val, smallest.test]))
        with self.assertRaises(ValueError):
            group_split_indices([0, 0, 1])
        with self.assertRaises(ValueError):
            group_split_indices([0, 1, 2, np.nan])

    def test_logistic_raw_coordinates_match_standardized_pipeline(self):
        C = 0.3
        probe = fit_logistic_probe(self.X, self.y, C=C, seed=5)
        reference = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, random_state=5, max_iter=3000)
        ).fit(self.X, self.y)
        # Include shifted unseen data; equivalence must not rely on centering it again.
        held_out = self.X[:15] + [0.03, 5, -2, 10]
        np.testing.assert_allclose(
            probe.decision_function(held_out), reference.decision_function(held_out), atol=1e-11
        )
        np.testing.assert_allclose(probe.predict_proba(held_out), reference.predict_proba(held_out))
        np.testing.assert_array_equal(probe.predict(held_out), reference.predict(held_out))

    def test_mean_difference_midpoint_is_translation_invariant(self):
        X = np.array([[8., 3], [10., 1], [12., 5], [14., 7]])
        y = np.array([0, 0, 1, 1])
        probe = fit_mean_difference(X, y)
        translated = fit_mean_difference(X + [300., -20], y)
        midpoint = (X[y == 0].mean(0) + X[y == 1].mean(0)) / 2
        self.assertAlmostEqual(probe.decision_function(midpoint[None])[0], 0)
        np.testing.assert_allclose(probe.direction, translated.direction)
        np.testing.assert_allclose(probe.decision_function(X), translated.decision_function(X + [300., -20]))

    def test_pca_uses_training_mean_for_holdout(self):
        X = np.array([[1., 2, 0], [3., 2, 0], [5., 2, 0]])
        pca = fit_train_pca(X, n_components=1)
        holdout = np.array([[103., 2, 0]])
        self.assertAlmostEqual(abs(pca.transform(holdout)[0, 0]), 100.)
        np.testing.assert_allclose(pca.transform(X).mean(0), 0, atol=1e-12)
        np.testing.assert_allclose(pca.mean, [3, 2, 0])

    def test_group_bootstrap_matches_whole_cluster_resampling(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        scores = np.array([-1, 1, 1, -1, -1, -1])
        groups = np.array([0, 0, 1, 1, 1, 2])
        seed, n_resamples = 22, 300
        rng, reference = np.random.default_rng(seed), []
        for _ in range(n_resamples):
            draw = rng.integers(3, size=3)
            rows = np.concatenate([np.flatnonzero(groups == group) for group in draw])
            reference.append(np.mean((scores[rows] >= 0) == y[rows]))
        actual = group_bootstrap_ci(y, scores, groups, "accuracy", n_resamples, seed)
        np.testing.assert_allclose(actual, np.quantile(reference, [.025, .975]))

    def test_metrics_use_scores_for_auc_and_fixed_threshold_for_accuracy(self):
        result = score_metrics([0, 0, 1, 1], [1, 2, 3, 4], threshold=2.5)
        self.assertEqual(result, {"auroc": 1., "balanced_accuracy": 1., "accuracy": 1.})
        self.assertTrue(np.isnan(score_metrics([1, 1], [1, 2])["auroc"]))
        self.assertTrue(np.isnan(score_metrics([1, 1], [1, 2])["balanced_accuracy"]))


class TinyBlock(nn.Module):
    def __init__(self, offset, as_tuple):
        super().__init__()
        self.register_buffer("offset", torch.tensor(offset, dtype=torch.float32))
        self.as_tuple = as_tuple

    def forward(self, hidden):
        hidden = hidden + self.offset
        return (hidden, None) if self.as_tuple else hidden


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(9, 3)
        self.embed_tokens.weight.data.copy_(torch.arange(27).reshape(9, 3) / 10)
        self.layers = nn.ModuleList([TinyBlock([1, 2, 3], True), TinyBlock([3, 2, 1], False)])
        self.fail = False

    def forward(self, input_ids, attention_mask, **_kwargs):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            output = block(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
        if self.fail:
            raise RuntimeError("deliberate decoder failure")
        return SimpleNamespace(last_hidden_state=hidden)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyDecoder()
        self.lm_head = nn.Linear(3, 9, bias=False)
        self.lm_head.weight.data.zero_()
        self.lm_head.weight.data[0, 0] = 1
        self.lm_head.weight.data[1, 1] = 1
        self.head_calls = 0

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, **inputs):
        hidden = self.model(**inputs).last_hidden_state
        self.head_calls += 1
        return SimpleNamespace(logits=self.lm_head(hidden))


class TransformerHelpersTest(unittest.TestCase):
    def setUp(self):
        self.model = TinyLM()
        self.ids = torch.tensor([[1, 2, 0, 0], [0, 0, 1, 2]])
        self.mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])

    def assertNoHooks(self):
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.model.layers))

    def test_last_valid_position_supports_left_right_and_interior_padding(self):
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]])
        torch.testing.assert_close(last_valid_token_indices(mask), torch.tensor([1, 3, 2]))
        with self.assertRaises(ValueError):
            last_valid_token_indices([[0, 0]])

    def test_extract_correct_block_and_token_without_lm_head(self):
        states = extract_last_token_states(self.model, self.ids, self.mask, [0, 1])
        expected = self.model.get_input_embeddings()(torch.tensor([2, 2])).detach()
        torch.testing.assert_close(states[0], expected + torch.tensor([1, 2, 3]))
        torch.testing.assert_close(states[1], expected + 4)
        self.assertEqual(states[0].dtype, torch.float32)
        self.assertEqual(states[0].device.type, "cpu")
        self.assertEqual(self.model.head_calls, 0)
        self.assertTrue(self.model.training)
        self.assertNoHooks()

    def test_extract_removes_hooks_and_restores_mixed_mode_after_failure(self):
        self.model.model.layers[0].eval()
        self.model.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            extract_last_token_states(self.model, self.ids, self.mask, [0, 1])
        self.assertNoHooks()
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.model.layers[0].training)
        self.assertTrue(self.model.model.layers[1].training)

    def test_raw_direction_intervention_has_expected_logit_effect(self):
        baseline = next_token_contrast(self.model, self.ids, self.mask, 0, 1)
        intervened = next_token_contrast(self.model, self.ids, self.mask, 0, 1,
                                        layer_index=0, direction=[3, 4, 0], strength=2)
        torch.testing.assert_close(intervened - baseline, torch.full((2,), -.4))
        repeated = next_token_contrast(self.model, self.ids, self.mask, 0, 1)
        torch.testing.assert_close(repeated, baseline)
        self.assertNoHooks()

    def test_intervention_hook_removed_after_forward_or_dimension_error(self):
        for fail, direction, error in [(True, [1, 0, 0], RuntimeError), (False, [1, 0], ValueError)]:
            self.model.model.fail = fail
            with self.assertRaises(error):
                next_token_contrast(self.model, self.ids, self.mask, 0, 1, 0, direction, 1)
            self.assertNoHooks()
            self.assertTrue(self.model.training)

    def test_attention_padding_invariance_and_gradients(self):
        probe = AttentionProbe(3)
        valid = torch.tensor([[[1., 2, 3], [4., 5, 6]]])
        padding = torch.full((1, 2, 3), 100_000.)
        right, left = torch.cat((valid, padding), 1), torch.cat((padding, valid), 1)
        right_output = probe(right, self.mask[:1])
        left_output = probe(left, self.mask[1:])
        torch.testing.assert_close(right_output, left_output)
        weights = probe.get_attention_weights(right, self.mask[:1])
        torch.testing.assert_close(weights, torch.tensor([[.5, .5, 0., 0.]]))
        right_output.sum().backward()
        self.assertIsNotNone(probe.query.grad)
        self.assertTrue(torch.isfinite(probe.query.grad).all())
        with self.assertRaises(ValueError):
            probe(right, torch.zeros((1, 4)))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from load_signature_analysis.grouped_mlp import (
    GroupEncoder,
    GroupedFingerprintMLP,
    MaskedGatedFusion,
)


class GroupEncoderTests(unittest.TestCase):
    """Verify stable encoding when individual features within a group are missing."""

    def test_partial_missing_values_are_safely_masked(self) -> None:
        encoder = GroupEncoder(
            feature_dimension=3,
            hidden_dimension=4,
            embedding_dimension=2,
            dropout_probability=0.0,
        )
        values = torch.tensor([[1.0, float("nan"), 3.0]])
        mask = torch.tensor([[1, 0, 1]])

        embedding = encoder(values, mask)

        self.assertEqual(tuple(embedding.shape), (1, 2))
        self.assertTrue(bool(torch.isfinite(embedding).all()))

    def test_multiple_hidden_layers_are_supported(self) -> None:
        encoder = GroupEncoder(
            feature_dimension=3,
            hidden_dimension=[6, 4],
            embedding_dimension=2,
            dropout_probability=0.0,
        )

        embedding = encoder(torch.ones(2, 3), torch.ones(2, 3))

        self.assertEqual(tuple(embedding.shape), (2, 2))
        self.assertEqual(encoder.hidden_dimensions, (6, 4))


class MaskedGatedFusionTests(unittest.TestCase):
    """Verify that missing groups receive no fusion weight."""

    def setUp(self) -> None:
        self.fusion = MaskedGatedFusion(("power", "temporal"), embedding_dimension=3)
        self.embeddings = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
            ]
        )

    def test_missing_group_weight_is_strictly_zero(self) -> None:
        _, weights = self.fusion(
            self.embeddings,
            torch.tensor([[1, 0], [1, 1]]),
        )

        self.assertEqual(weights[0, 1].item(), 0.0)
        self.assertEqual(weights[0, 0].item(), 1.0)
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2)))

    def test_temporal_only_has_unit_weight(self) -> None:
        _, weights = self.fusion(
            self.embeddings[:1],
            torch.tensor([[0, 1]]),
        )

        self.assertEqual(weights[0, 0].item(), 0.0)
        self.assertEqual(weights[0, 1].item(), 1.0)

    def test_all_groups_missing_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "全组缺失"):
            self.fusion(self.embeddings[:1], torch.tensor([[0, 0]]))


class GroupedFingerprintMLPTests(unittest.TestCase):
    """Verify flat-input grouping, structured dropout and backpropagation."""

    @staticmethod
    def _model(**overrides: object) -> GroupedFingerprintMLP:
        parameters: dict[str, object] = {
            "group_dimensions": {"power": 3, "temporal": 2},
            # Use interleaved indices to verify that feature groups need not be contiguous in the input.
            "group_slices": {"power": [0, 2, 4], "temporal": [1, 3]},
            "encoder_hidden_dimension": 8,
            "embedding_dimension": 4,
            "classifier_hidden_dimension": 4,
            "dropout_probability": 0.0,
            "feature_dropout_enabled": False,
        }
        parameters.update(overrides)
        return GroupedFingerprintMLP(**parameters)

    def test_flat_input_returns_expected_shapes(self) -> None:
        model = self._model().eval()
        values = torch.randn(3, 5)
        masks = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
            ]
        )

        logits, weights = model(values=values, masks=masks)

        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(weights.shape), (3, 2))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(3)))
        self.assertEqual(weights[1, 1].item(), 0.0)
        self.assertEqual(weights[1, 0].item(), 1.0)
        self.assertEqual(weights[2, 0].item(), 0.0)
        self.assertEqual(weights[2, 1].item(), 1.0)

    def test_all_features_missing_raises_clear_error(self) -> None:
        model = self._model().eval()
        with self.assertRaisesRegex(ValueError, "全组缺失"):
            model(torch.zeros(1, 5), torch.zeros(1, 5))

    def test_structured_dropout_never_restores_natural_missing_or_empties(self) -> None:
        model = self._model(
            feature_dropout_enabled=True,
            group_dropout_probability=1.0,
            feature_dropout_probability=1.0,
        ).train()
        natural_mask = torch.tensor(
            [
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        generator = torch.Generator().manual_seed(7)

        effective_mask = model.apply_structured_feature_dropout(
            natural_mask,
            generator=generator,
        )

        self.assertFalse(bool((effective_mask & ~natural_mask).any()))
        self.assertTrue(bool(effective_mask.any(dim=1).all()))

    def test_structured_dropout_is_disabled_during_evaluation(self) -> None:
        model = self._model(
            feature_dropout_enabled=True,
            group_dropout_probability=1.0,
            feature_dropout_probability=1.0,
        ).eval()
        natural_mask = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.bool)

        effective_mask = model.apply_structured_feature_dropout(natural_mask)

        self.assertTrue(torch.equal(effective_mask, natural_mask))

    def test_backward_pass_produces_finite_gradients(self) -> None:
        torch.manual_seed(11)
        model = self._model().train()
        values = torch.randn(4, 5)
        masks = torch.ones(4, 5)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

        logits, _ = model(values, masks)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))

    def test_default_contiguous_slices_support_27_features(self) -> None:
        model = GroupedFingerprintMLP(
            {"power": 17, "temporal": 10},
            dropout_probability=0.0,
            feature_dropout_enabled=False,
        ).eval()

        logits, weights = model(torch.randn(2, 27), torch.ones(2, 27))

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(weights.shape), (2, 2))
        self.assertEqual(model.group_order, ("power", "temporal"))

    def test_multiple_hidden_layers_keep_forward_and_backward_contract(self) -> None:
        model = self._model(
            encoder_hidden_dimension=[8, 6],
            classifier_hidden_dimension=[6, 4],
        ).train()
        values = torch.randn(4, 5)
        masks = torch.ones(4, 5)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

        logits, gates = model(values, masks)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()

        self.assertEqual(tuple(logits.shape), (4,))
        self.assertEqual(tuple(gates.shape), (4, 2))
        self.assertEqual(model.encoder_hidden_dimensions, (8, 6))
        self.assertEqual(model.classifier_hidden_dimensions, (6, 4))
        self.assertTrue(
            all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )

    def test_single_dimension_list_is_checkpoint_compatible_with_scalar(self) -> None:
        scalar_model = self._model().eval()
        list_model = self._model(
            encoder_hidden_dimension=[8],
            classifier_hidden_dimension=[4],
        ).eval()

        list_model.load_state_dict(scalar_model.state_dict(), strict=True)
        values = torch.randn(2, 5)
        masks = torch.ones(2, 5)
        scalar_logits, scalar_gates = scalar_model(values, masks)
        list_logits, list_gates = list_model(values, masks)

        self.assertTrue(torch.allclose(scalar_logits, list_logits))
        self.assertTrue(torch.allclose(scalar_gates, list_gates))


if __name__ == "__main__":
    unittest.main()

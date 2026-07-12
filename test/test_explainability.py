"""Tests for explainability module.

Following TDD approach: tests first, then implementation.
Tests focus on the decoupled interface and the DiCE adapter.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# === Fixtures ===


@pytest.fixture
def sample_data():
    """Sample data for explanation."""
    X = pd.DataFrame(
        {
            "Bacteroides": [0.10, 0.20, 0.15, 0.25],
            "Prevotella": [0.05, 0.10, 0.00, 0.08],
            "Firmicutes": [0.85, 0.70, 0.85, 0.67],
        },
        index=["S1", "S2", "S3", "S4"],
    )
    y = pd.Series([0, 1, 0, 1], index=["S1", "S2", "S3", "S4"], name="disease")
    return X, y


@pytest.fixture
def mock_model():
    """Mock trained model."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0, 1])
    return model


# === Interface Tests ===


class TestExplainerInterface:
    """Test the BaseExplainer interface."""

    def test_base_explainer_structure(self):
        """BaseExplainer should be an abstract base class."""
        from microfactual.explainability.base import BaseExplainer

        # Should rely on abc
        assert hasattr(BaseExplainer, "__abstractmethods__")

        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            BaseExplainer(model=None, data=None)


# === DiCE Adapter Tests ===


class TestDiCEExplainer:
    """Test the DiCE adapter implementation."""

    def test_initialization(self, sample_data, mock_model):
        """Can initialize DiCEExplainer."""
        from microfactual.explainability.counterfactuals import DiCEExplainer

        X, y = sample_data

        # Mock dice_ml to avoid actual dependency during unit test initialization if possible,
        # or just ensure it constructs correctly.
        with patch("microfactual.explainability.counterfactuals.dice_ml"):
            explainer = DiCEExplainer(
                model=mock_model,
                background_data=X,
                target_column="disease",
                target_data=y,
            )
            assert explainer is not None

    def test_generate_counterfactuals(self, sample_data, mock_model):
        """generate_counterfactuals calls underlying DiCE backend."""
        from microfactual.explainability.counterfactuals import DiCEExplainer

        X, y = sample_data
        query_instance = X.iloc[[0]]

        with patch("microfactual.explainability.counterfactuals.dice_ml") as mock_dice:
            # Setup mocks
            mock_exp_instance = MagicMock()
            mock_dice.Dice.return_value = mock_exp_instance

            explainer = DiCEExplainer(
                model=mock_model,
                background_data=X,
                target_column="disease",
                target_data=y,
            )

            # Act
            explainer.explain(query_instance)

            # Assert
            mock_exp_instance.generate_counterfactuals.assert_called_once()


# === explain_counterfactual() convenience API ===


class TestExplainCounterfactual:
    """Test the top-level one-call counterfactual entry point."""

    def test_wires_through_to_dice(self, sample_data, mock_model):
        """explain_counterfactual builds an explainer and generates CFs."""
        from microfactual.explainability.counterfactuals import explain_counterfactual

        X, y = sample_data
        query = X.iloc[[0]]

        with patch("microfactual.explainability.counterfactuals.dice_ml") as mock_dice:
            mock_exp_instance = MagicMock()
            mock_dice.Dice.return_value = mock_exp_instance

            explain_counterfactual(
                model=mock_model,
                query=query,
                background_data=X,
                y=y,
                total_CFs=3,
                features_to_vary=["Bacteroides"],
                return_raw=True,  # skip result wrapping for this wiring test
            )

            mock_exp_instance.generate_counterfactuals.assert_called_once()
            _, called_kwargs = mock_exp_instance.generate_counterfactuals.call_args
            assert called_kwargs["total_CFs"] == 3
            assert called_kwargs["desired_class"] == "opposite"
            assert called_kwargs["features_to_vary"] == ["Bacteroides"]

    def test_raises_without_dice(self, sample_data, mock_model):
        """A clear ImportError is raised when dice-ml is unavailable."""
        from microfactual.explainability.counterfactuals import explain_counterfactual

        X, y = sample_data

        with patch("microfactual.explainability.counterfactuals.dice_ml", None):
            with pytest.raises(ImportError, match="explainability"):
                explain_counterfactual(
                    model=mock_model, query=X.iloc[[0]], background_data=X, y=y
                )

    def test_exported_at_top_level(self):
        """explain_counterfactual is part of the public API."""
        import microfactual as mf

        assert hasattr(mf, "explain_counterfactual")
        assert "explain_counterfactual" in mf.__all__

    def test_drops_non_flipping_counterfactuals(self, sample_data):
        """DiCE rows that don't flip the prediction are discarded."""
        from microfactual.explainability import counterfactuals as cf_mod

        X, y = sample_data
        model = _ThresholdModel()  # predicts 1 iff feature 0 > 0.5
        query = pd.DataFrame([[0.0, 0.0, 0.0]], columns=X.columns)  # predicts 0

        # One row flips (feature 0 = 1.0 -> class 1), one does not (stays class 0).
        final = pd.DataFrame(
            [[1.0, 0.0, 0.0], [0.2, 9.0, 9.0]], columns=list(X.columns)
        )
        example = MagicMock()
        example.final_cfs_df = final
        raw = MagicMock()
        raw.cf_examples_list = [example]

        with patch.object(cf_mod, "DiCEExplainer") as MockExplainer:
            MockExplainer.return_value.explain.return_value = raw
            result = cf_mod.explain_counterfactual(model, query, X, y, sparse=False)

        assert result.n_counterfactuals == 1  # only the flipping row survives
        assert result.validity == 1.0

    def test_returns_single_result_when_dice_finds_nothing(self, sample_data):
        """Empty cf_examples_list -> one empty CounterfactualResult, not a list."""
        from microfactual.explainability import counterfactuals as cf_mod
        from microfactual.explainability.result import CounterfactualResult

        X, y = sample_data
        model = _ThresholdModel()
        raw = MagicMock()
        raw.cf_examples_list = []  # DiCE returned no examples

        with patch.object(cf_mod, "DiCEExplainer") as MockExplainer:
            MockExplainer.return_value.explain.return_value = raw
            result = cf_mod.explain_counterfactual(model, X.iloc[[0]], X, y)

        # Must be a single (empty) result, never a bare list (which broke callers
        # with AttributeError: 'list' object has no attribute 'changes').
        assert isinstance(result, CounterfactualResult)
        assert result.n_counterfactuals == 0
        assert result.changes().empty  # does not raise on pd.concat([])
        assert result.changes(0).empty
        assert result.n_changes == []


# === Sparsification & result object (no DiCE required) ===


class _ThresholdModel:
    """Toy classifier: predicts class 1 iff feature 0 exceeds 0.5."""

    def predict(self, X):
        X = np.asarray(X)
        return (X[:, 0] > 0.5).astype(int)


class TestSparsify:
    """The greedy post-hoc sparsifier."""

    def test_reverts_irrelevant_changes(self):
        from microfactual.explainability.result import sparsify_counterfactual

        model = _ThresholdModel()
        original = np.array([0.0, 0.0, 0.0, 0.0])
        counterfactual = np.array([1.0, 1.0, 1.0, 1.0])  # all changed, flips to 1

        sparse = sparsify_counterfactual(original, counterfactual, model)

        # Only feature 0 matters for the flip; the rest revert to original.
        assert sparse[0] == 1.0
        assert (sparse[1:] == 0.0).all()

    def test_preserves_prediction(self):
        from microfactual.explainability.result import sparsify_counterfactual

        model = _ThresholdModel()
        original = np.array([0.0, 0.2, 0.9])
        counterfactual = np.array([1.0, 0.8, 0.1])

        sparse = sparsify_counterfactual(original, counterfactual, model)
        assert model.predict(sparse.reshape(1, -1))[0] == 1


class TestCounterfactualResult:
    """The interpretable result wrapper."""

    def _make(self):
        from microfactual.explainability.result import CounterfactualResult

        model = _ThresholdModel()
        return CounterfactualResult(
            original=[0.0, 0.0, 0.0],
            counterfactuals=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            model=model,
            feature_names=["f0", "f1", "f2"],
            class_names=["Healthy", "Disease"],
        )

    def test_n_changes(self):
        res = self._make()
        assert res.n_changes == [1, 1]

    def test_validity(self):
        # cf0 ([1,0,0]) flips to 1; cf1 ([0,0,1]) does not -> 0.5
        res = self._make()
        assert res.validity == 0.5

    def test_changes_table(self):
        res = self._make()
        changes = res.changes(0)
        assert list(changes["feature"]) == ["f0"]
        assert changes.iloc[0]["direction"] == "increase"
        assert changes.iloc[0]["delta"] == 1.0

    def test_summary_uses_class_names(self):
        res = self._make()
        assert "Healthy" in res.summary()
        assert "Disease" in res.summary()

    def test_exported_at_top_level(self):
        import microfactual as mf

        assert hasattr(mf, "CounterfactualResult")
        assert "CounterfactualResult" in mf.__all__


class TestCounterfactualImportance:
    """Cohort-level aggregation of counterfactuals."""

    def test_aggregates_across_cohort(self):
        from microfactual.explainability import counterfactuals as cf_mod
        from microfactual.explainability.result import CounterfactualResult

        model = _ThresholdModel()

        def result(cfs):
            return CounterfactualResult([0.0, 0.0], cfs, model, ["f0", "f1"])

        # Two samples change f0 (increase); one changes f1 (decrease).
        canned = [
            result([[1.0, 0.0]]),
            result([[1.0, 0.0]]),
            result([[0.0, -1.0]]),
        ]
        X = pd.DataFrame({"f0": [0.0, 0.0, 0.0], "f1": [0.0, 0.0, 0.0]})

        with patch.object(cf_mod, "explain_counterfactual", side_effect=canned):
            imp = cf_mod.counterfactual_importance(model, X, y=[0, 1, 0])

        f0 = imp[imp["feature"] == "f0"].iloc[0]
        assert f0["n_samples"] == 2
        assert f0["frequency"] == pytest.approx(2 / 3)
        assert f0["direction"] == "increase"
        f1 = imp[imp["feature"] == "f1"].iloc[0]
        assert f1["n_samples"] == 1
        assert f1["direction"] == "decrease"

    def test_exported_at_top_level(self):
        import microfactual as mf

        assert hasattr(mf, "counterfactual_importance")
        assert "counterfactual_importance" in mf.__all__

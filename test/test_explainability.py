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

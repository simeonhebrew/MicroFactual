"""Tests for visualization module.

Following TDD approach: tests first, then implementation.
"""

import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# === Fixtures ===


@pytest.fixture
def binary_predictions():
    """Sample binary classification results."""
    return {
        "y_true": pd.Series([0, 1, 0, 1, 0, 1, 0, 1]),
        "y_pred": pd.Series([0, 1, 0, 0, 0, 1, 1, 1]),
        "y_proba": pd.Series([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7]),
    }


@pytest.fixture
def feature_importance_data():
    """Sample feature importance scores."""
    return pd.Series(
        [0.35, 0.25, 0.20, 0.12, 0.08],
        index=[
            "Bacteroides",
            "Firmicutes",
            "Prevotella",
            "Fusobacterium",
            "Akkermansia",
        ],
    )


# === ROC Curve Tests ===


class TestPlotROC:
    """Test ROC curve plotting."""

    def test_plot_roc_returns_figure(self, binary_predictions):
        """plot_roc returns a matplotlib figure."""
        from microfactual.visualization import plot_roc

        fig = plot_roc(
            binary_predictions["y_true"],
            binary_predictions["y_proba"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_roc_shows_auc(self, binary_predictions):
        """ROC plot includes AUC in legend."""
        from microfactual.visualization import plot_roc

        fig = plot_roc(
            binary_predictions["y_true"],
            binary_predictions["y_proba"],
        )

        ax = fig.axes[0]
        legend_text = ax.get_legend().get_texts()[0].get_text()
        assert "AUC" in legend_text
        plt.close(fig)


# === Confusion Matrix Tests ===


class TestPlotConfusionMatrix:
    """Test confusion matrix plotting."""

    def test_plot_confusion_matrix_returns_figure(self, binary_predictions):
        """plot_confusion_matrix returns a matplotlib figure."""
        from microfactual.visualization import plot_confusion_matrix

        fig = plot_confusion_matrix(
            binary_predictions["y_true"],
            binary_predictions["y_pred"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_confusion_matrix_with_labels(self, binary_predictions):
        """Confusion matrix can use custom class labels."""
        from microfactual.visualization import plot_confusion_matrix

        fig = plot_confusion_matrix(
            binary_predictions["y_true"],
            binary_predictions["y_pred"],
            labels=["Healthy", "Disease"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# === Feature Importance Tests ===


class TestPlotFeatureImportance:
    """Test feature importance plotting."""

    def test_plot_feature_importance_returns_figure(self, feature_importance_data):
        """plot_feature_importance returns a matplotlib figure."""
        from microfactual.visualization import plot_feature_importance

        fig = plot_feature_importance(feature_importance_data)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_feature_importance_top_n(self, feature_importance_data):
        """Can limit to top N features."""
        from microfactual.visualization import plot_feature_importance

        fig = plot_feature_importance(feature_importance_data, top_n=3)

        ax = fig.axes[0]
        # Should only show 3 bars
        assert len(ax.patches) == 3
        plt.close(fig)

    def test_plot_feature_importance_from_model(self, feature_importance_data):
        """Can extract importance from fitted model."""
        from unittest.mock import Mock

        from microfactual.visualization import plot_feature_importance

        # Mock a model with feature_importances_
        mock_model = Mock()
        mock_model.feature_importances_ = feature_importance_data.values

        fig = plot_feature_importance(
            mock_model,
            feature_names=feature_importance_data.index.tolist(),
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

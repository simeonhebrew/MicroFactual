"""Tests for ExplainerDashboard wrapper.

Tests that we can launch the dashboard with our models.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# === Fixtures ===

@pytest.fixture
def sample_data():
    """Sample data."""
    X = pd.DataFrame(
        {
            "F1": [0.1, 0.2, 0.3, 0.4],
            "F2": [1, 2, 3, 4]
        },
        index=["S1", "S2", "S3", "S4"]
    )
    y = pd.Series([0, 1, 0, 1])
    return X, y

# === Dashboard Tests ===

class TestLaunchDashboard:
    """Test standard launch_dashboard function."""

    def test_launches_with_sklearn_model(self, sample_data):
        """Standard usage with sklearn model."""
        from microfactual.visualization.dashboard import launch_dashboard
        X, y = sample_data
        model = MagicMock()
        
        # We need to mock explainerdashboard imports to avoid actual launch
        with patch("microfactual.visualization.dashboard.ClassifierExplainer") as MockCx, \
             patch("microfactual.visualization.dashboard.ExplainerDashboard") as MockDb:
            
            db_instance = MagicMock()
            MockDb.return_value = db_instance
            
            # Act
            result = launch_dashboard(model, X, y, run=False)
            
            # Assert
            MockCx.assert_called_once()
            MockDb.assert_called_once()
            assert result == db_instance

    def test_clean_feature_names(self, sample_data):
        """Should clean special characters in column names."""
        from microfactual.visualization.dashboard import launch_dashboard
        X, y = sample_data
        X.columns = ["F.1", "F{2}"] # Bad names for dashboard
        model = MagicMock()
        
        with patch("microfactual.visualization.dashboard.ClassifierExplainer") as MockCx, \
             patch("microfactual.visualization.dashboard.ExplainerDashboard"):
            
            launch_dashboard(model, X, y, run=False)
            
            # Check that X passed to explainer has sanitized columns
            # The args to ClassifierExplainer are (model, X, y, ...)
            args, _ = MockCx.call_args
            X_passed = args[1]
            assert "F_1" in X_passed.columns
            assert "F_2_" in X_passed.columns

    def test_extract_model_from_microbiome_classifier(self, sample_data):
        """Should handle MicrobiomeClassifier wrapper correctly."""
        from microfactual.visualization.dashboard import launch_dashboard
        from microfactual.models.classifiers import MicrobiomeClassifier
        
        X, y = sample_data
        mf_model = MagicMock(spec=MicrobiomeClassifier)
        # Mock inner pipeline
        inner_pipe = MagicMock()
        mf_model.pipeline_ = inner_pipe
        
        with patch("microfactual.visualization.dashboard.ClassifierExplainer") as MockCx, \
             patch("microfactual.visualization.dashboard.ExplainerDashboard"):
             
             launch_dashboard(mf_model, X, y, run=False)
             
             # Should pass the inner pipeline or the model itself usually
             # But let's see how our implementation handles it.
             # Ideally it should just pass the model if it supports predict_proba
             args, _ = MockCx.call_args
             assert args[0] == mf_model

"""Tests for sklearn-compatible preprocessing transforms.

Following TDD approach: write tests first, then implementation.
These transforms should work with sklearn.Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


# === Fixtures ===


@pytest.fixture
def abundance_matrix():
    """Abundance data in sklearn format (samples x features)."""
    return pd.DataFrame(
        {
            "Bacteroides": [0.10, 0.20, 0.15, 0.25, 0.12],
            "Prevotella": [0.05, 0.10, 0.00, 0.08, 0.03],
            "Fusobacterium": [0.00, 0.00, 0.05, 0.02, 0.00],
            "Firmicutes": [0.85, 0.70, 0.80, 0.65, 0.85],
        },
        index=["S1", "S2", "S3", "S4", "S5"],
    )


# === AbundanceFilter Tests ===


class TestAbundanceFilter:
    """Test abundance-based feature filtering."""

    def test_filters_low_abundance_features(self, abundance_matrix):
        """Features below threshold are removed."""
        from microfactual.preprocessing.transforms import AbundanceFilter

        filt = AbundanceFilter(min_abundance=0.1)
        result = filt.fit_transform(abundance_matrix)

        # Should keep Bacteroides and Firmicutes (mean > 0.1)
        assert "Firmicutes" in result.columns
        assert "Bacteroides" in result.columns
        # Should remove low abundance features
        assert result.shape[1] < abundance_matrix.shape[1]

    def test_is_sklearn_compatible(self, abundance_matrix):
        """Works in sklearn Pipeline."""
        from microfactual.preprocessing.transforms import AbundanceFilter

        pipe = Pipeline([("filter", AbundanceFilter(min_abundance=0.1))])
        result = pipe.fit_transform(abundance_matrix)

        assert result is not None
        assert len(result) == len(abundance_matrix)


# === PrevalenceFilter Tests ===


class TestPrevalenceFilter:
    """Test prevalence-based feature filtering."""

    def test_filters_rare_features(self, abundance_matrix):
        """Features present in few samples are removed."""
        from microfactual.preprocessing.transforms import PrevalenceFilter

        filt = PrevalenceFilter(min_prevalence=0.5)
        result = filt.fit_transform(abundance_matrix)

        # Fusobacterium only present in 2/5 samples (40%) - should be removed
        assert "Fusobacterium" not in result.columns
        # Bacteroides present in all samples - should be kept
        assert "Bacteroides" in result.columns

    def test_is_sklearn_compatible(self, abundance_matrix):
        """Works in sklearn Pipeline."""
        from microfactual.preprocessing.transforms import PrevalenceFilter

        pipe = Pipeline([("filter", PrevalenceFilter(min_prevalence=0.5))])
        result = pipe.fit_transform(abundance_matrix)

        assert result is not None


# === CLRTransform Tests ===


class TestCLRTransform:
    """Test centered log-ratio transformation."""

    def test_transforms_data(self, abundance_matrix):
        """CLR produces valid transformed values."""
        from microfactual.preprocessing.transforms import CLRTransform

        clr = CLRTransform()
        result = clr.fit_transform(abundance_matrix)

        # Output should have same shape
        assert result.shape == abundance_matrix.shape
        # CLR should center the data (mean of each row close to 0)
        row_means = result.mean(axis=1)
        assert all(abs(m) < 1e-10 for m in row_means)

    def test_handles_zeros(self, abundance_matrix):
        """Handles zero values via pseudocount."""
        from microfactual.preprocessing.transforms import CLRTransform

        clr = CLRTransform(pseudocount=1e-6)
        result = clr.fit_transform(abundance_matrix)

        # Should not have NaN or inf
        assert not result.isnull().any().any()
        assert not np.isinf(result).any().any()

    def test_is_sklearn_compatible(self, abundance_matrix):
        """Works in sklearn Pipeline."""
        from microfactual.preprocessing.transforms import CLRTransform

        pipe = Pipeline([("clr", CLRTransform())])
        result = pipe.fit_transform(abundance_matrix)

        assert result is not None
        assert result.shape == abundance_matrix.shape


# === Combined Pipeline Tests ===


class TestPreprocessingPipeline:
    """Test combining transforms in a pipeline."""

    def test_full_preprocessing_pipeline(self, abundance_matrix):
        """Full filter -> CLR pipeline works."""
        from microfactual.preprocessing.transforms import (
            AbundanceFilter,
            PrevalenceFilter,
            CLRTransform,
        )

        pipe = Pipeline(
            [
                ("abundance", AbundanceFilter(min_abundance=0.05)),
                ("prevalence", PrevalenceFilter(min_prevalence=0.5)),
                ("clr", CLRTransform()),
            ]
        )

        result = pipe.fit_transform(abundance_matrix)

        # Should have reduced features and transformed values
        assert result.shape[0] == abundance_matrix.shape[0]  # Same samples
        assert result.shape[1] <= abundance_matrix.shape[1]  # Fewer or equal features

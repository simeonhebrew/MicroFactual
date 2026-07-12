"""Tests for MicrobiomeClassifier and high-level API.

Following TDD approach: tests first, then implementation.
"""

import numpy as np
import pandas as pd
import pytest

# === Fixtures ===


@pytest.fixture
def sample_X():
    """Sample feature matrix (samples x features)."""
    return pd.DataFrame(
        {
            "Bacteroides": [0.10, 0.20, 0.15, 0.25],
            "Prevotella": [0.05, 0.10, 0.00, 0.08],
            "Firmicutes": [0.85, 0.70, 0.85, 0.67],
        },
        index=["S1", "S2", "S3", "S4"],
    )


@pytest.fixture
def sample_y():
    """Sample target vector."""
    return pd.Series([0, 1, 0, 1], index=["S1", "S2", "S3", "S4"], name="disease")


# === MicrobiomeClassifier Tests ===


class TestMicrobiomeClassifier:
    """Test the main classifier wrapper."""

    def test_basic_fit_predict(self, sample_X, sample_y):
        """Can fit and predict with default settings."""
        from microfactual.models.classifiers import MicrobiomeClassifier

        clf = MicrobiomeClassifier()
        clf.fit(sample_X, sample_y)

        predictions = clf.predict(sample_X)
        assert len(predictions) == len(sample_y)

    def test_predict_proba(self, sample_X, sample_y):
        """Returns probability predictions."""
        from microfactual.models.classifiers import MicrobiomeClassifier

        clf = MicrobiomeClassifier()
        clf.fit(sample_X, sample_y)

        proba = clf.predict_proba(sample_X)
        assert proba.shape[0] == len(sample_y)
        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_different_algorithms(self, sample_X, sample_y):
        """Can use different algorithms."""
        from microfactual.models.classifiers import MicrobiomeClassifier

        for algorithm in ["random_forest", "logistic"]:
            clf = MicrobiomeClassifier(algorithm=algorithm)
            clf.fit(sample_X, sample_y)
            predictions = clf.predict(sample_X)
            assert len(predictions) == len(sample_y)

    def test_with_preprocessing(self, sample_X, sample_y):
        """Works with built-in preprocessing."""
        from microfactual.models.classifiers import MicrobiomeClassifier

        clf = MicrobiomeClassifier(preprocessing="auto")
        clf.fit(sample_X, sample_y)

        predictions = clf.predict(sample_X)
        assert len(predictions) == len(sample_y)

    def test_is_sklearn_compatible(self, sample_X, sample_y):
        """Works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        from microfactual.models.classifiers import MicrobiomeClassifier

        clf = MicrobiomeClassifier(preprocessing=None)
        # With only 4 samples, use 2-fold CV
        scores = cross_val_score(clf, sample_X, sample_y, cv=2)

        assert len(scores) == 2

    def test_get_params_exposes_forwarded_model_params(self):
        """Forwarded model kwargs appear in get_params (sklearn contract)."""
        from microfactual.models.classifiers import MicrobiomeClassifier

        clf = MicrobiomeClassifier(algorithm="random_forest", n_estimators=42)
        params = clf.get_params()
        assert params["algorithm"] == "random_forest"
        assert params["n_estimators"] == 42

    def test_set_params_roundtrip(self):
        """set_params can update both declared and forwarded params."""
        from microfactual.models.classifiers import MicrobiomeClassifier

        clf = MicrobiomeClassifier()
        clf.set_params(algorithm="logistic", n_estimators=10, preprocessing=None)
        params = clf.get_params()
        assert params["algorithm"] == "logistic"
        assert params["preprocessing"] is None
        assert params["n_estimators"] == 10

    def test_clone_preserves_forwarded_params(self):
        """sklearn.clone round-trips forwarded model params."""
        from sklearn.base import clone

        from microfactual.models.classifiers import MicrobiomeClassifier

        clf = MicrobiomeClassifier(algorithm="random_forest", n_estimators=33)
        cloned = clone(clf)
        assert cloned.get_params()["n_estimators"] == 33

    def test_gridsearch_over_forwarded_param(self, sample_X, sample_y):
        """GridSearchCV can tune a forwarded underlying-model param."""
        from sklearn.model_selection import GridSearchCV

        from microfactual.models.classifiers import MicrobiomeClassifier

        grid = GridSearchCV(
            MicrobiomeClassifier(algorithm="random_forest", preprocessing=None),
            param_grid={"n_estimators": [5, 10]},
            cv=2,
        )
        grid.fit(sample_X, sample_y)
        assert grid.best_params_["n_estimators"] in (5, 10)


# === High-Level API Tests ===


class TestHighLevelAPI:
    """Test the simple mf.classify() function."""

    def test_classify_from_files(self, tmp_path):
        """classify() works with file paths."""
        # Create temp files
        abundance = pd.DataFrame(
            {
                "S1": [0.1, 0.9],
                "S2": [0.2, 0.8],
                "S3": [0.15, 0.85],
                "S4": [0.25, 0.75],
            },
            index=["Bacteroides", "Firmicutes"],
        )
        metadata = pd.DataFrame(
            {
                "Sample ID": ["S1", "S2", "S3", "S4"],
                "disease": ["healthy", "CRC", "healthy", "CRC"],
            }
        )

        abundance_file = tmp_path / "abundance.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        abundance.to_csv(abundance_file, sep="\t", index_label="Species")
        metadata.to_csv(metadata_file, sep="\t", index=False)

        import microfactual as mf

        results = mf.classify(
            str(abundance_file),
            str(metadata_file),
            target_column="disease",
            cv_folds=2,
        )

        assert "model" in results
        assert "cv_scores" in results
        assert "test_accuracy" in results["cv_scores"]

    def test_classify_returns_trained_model(self, tmp_path):
        """classify() returns a fitted model."""
        # Create temp files
        abundance = pd.DataFrame(
            {
                "S1": [0.1, 0.9],
                "S2": [0.2, 0.8],
                "S3": [0.15, 0.85],
                "S4": [0.25, 0.75],
            },
            index=["Bacteroides", "Firmicutes"],
        )
        metadata = pd.DataFrame(
            {
                "Sample ID": ["S1", "S2", "S3", "S4"],
                "disease": ["healthy", "CRC", "healthy", "CRC"],
            }
        )

        abundance_file = tmp_path / "abundance.tsv"
        metadata_file = tmp_path / "metadata.tsv"
        abundance.to_csv(abundance_file, sep="\t", index_label="Species")
        metadata.to_csv(metadata_file, sep="\t", index=False)

        import microfactual as mf

        results = mf.classify(
            str(abundance_file),
            str(metadata_file),
            target_column="disease",
            cv_folds=2,
        )

        # Model should be fitted
        model = results["model"]
        # Can make predictions
        X = abundance.T
        predictions = model.predict(X)
        assert len(predictions) == 4

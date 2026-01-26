"""Tests for MicrobiomeDataset - the central data abstraction.

Following TDD approach: write tests first, then implementation.
Tests are domain-focused and simple.
"""

import pandas as pd
import pytest

# === Fixtures ===


@pytest.fixture
def sample_abundance():
    """Realistic microbiome abundance data (species x samples)."""
    return pd.DataFrame(
        {
            "Sample1": [0.10, 0.05, 0.00, 0.85],
            "Sample2": [0.20, 0.10, 0.00, 0.70],
            "Sample3": [0.15, 0.00, 0.05, 0.80],
            "Sample4": [0.25, 0.08, 0.02, 0.65],
        },
        index=["Bacteroides", "Prevotella", "Fusobacterium", "Firmicutes"],
    )


@pytest.fixture
def sample_metadata():
    """Sample metadata with disease status."""
    return pd.DataFrame(
        {
            "Sample ID": ["Sample1", "Sample2", "Sample3", "Sample4"],
            "disease": ["healthy", "CRC", "healthy", "CRC"],
            "age": [45, 52, 38, 61],
        }
    )


# === Dataset Creation Tests ===


class TestMicrobiomeDatasetCreation:
    """Test dataset creation and initialization."""

    def test_create_from_dataframes(self, sample_abundance, sample_metadata):
        """Can create dataset from pandas DataFrames."""
        from microfactual.core.dataset import MicrobiomeDataset

        dataset = MicrobiomeDataset(
            abundance=sample_abundance,
            metadata=sample_metadata,
            target_column="disease",
        )

        assert dataset is not None
        assert len(dataset.feature_names) == 4
        assert len(dataset.sample_names) == 4

    def test_create_from_files(self, tmp_path, sample_abundance, sample_metadata):
        """Can create dataset from TSV files."""
        # Save fixtures to temp files
        abundance_file = tmp_path / "abundance.tsv"
        metadata_file = tmp_path / "metadata.tsv"

        sample_abundance.to_csv(abundance_file, sep="\t", index_label="Species")
        sample_metadata.to_csv(metadata_file, sep="\t", index=False)

        from microfactual.core.dataset import MicrobiomeDataset

        dataset = MicrobiomeDataset.from_files(
            str(abundance_file),
            str(metadata_file),
            target_column="disease",
        )

        assert len(dataset.feature_names) == 4
        assert len(dataset.sample_names) == 4


# === Sklearn Compatibility Tests ===


class TestSklearnCompatibility:
    """Test sklearn-compatible X and y properties."""

    def test_X_is_samples_by_features(self, sample_abundance, sample_metadata):
        """X property returns samples x features matrix."""
        from microfactual.core.dataset import MicrobiomeDataset

        dataset = MicrobiomeDataset(
            abundance=sample_abundance,
            metadata=sample_metadata,
            target_column="disease",
        )

        X = dataset.X

        # Should be transposed: samples as rows
        assert X.shape == (4, 4)  # 4 samples, 4 features
        assert list(X.index) == ["Sample1", "Sample2", "Sample3", "Sample4"]
        assert list(X.columns) == [
            "Bacteroides",
            "Prevotella",
            "Fusobacterium",
            "Firmicutes",
        ]

    def test_y_is_encoded_labels(self, sample_abundance, sample_metadata):
        """Y property returns encoded target vector."""
        from microfactual.core.dataset import MicrobiomeDataset

        dataset = MicrobiomeDataset(
            abundance=sample_abundance,
            metadata=sample_metadata,
            target_column="disease",
        )

        y = dataset.y

        assert len(y) == 4
        # Should be numeric codes (0 or 1 for binary)
        assert set(y.unique()).issubset({0, 1})

    def test_works_with_sklearn_cross_validate(self, sample_abundance, sample_metadata):
        """Dataset X, y work directly with sklearn."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        from microfactual.core.dataset import MicrobiomeDataset

        dataset = MicrobiomeDataset(
            abundance=sample_abundance,
            metadata=sample_metadata,
            target_column="disease",
        )

        # Should work without errors
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scores = cross_val_score(model, dataset.X, dataset.y, cv=2)

        assert len(scores) == 2


# === Dataset Information Tests ===


class TestDatasetInfo:
    """Test dataset metadata and info methods."""

    def test_get_info_returns_stats(self, sample_abundance, sample_metadata):
        """get_info returns useful statistics."""
        from microfactual.core.dataset import MicrobiomeDataset

        dataset = MicrobiomeDataset(
            abundance=sample_abundance,
            metadata=sample_metadata,
            target_column="disease",
        )

        info = dataset.get_info()

        assert info["n_samples"] == 4
        assert info["n_features"] == 4
        assert "healthy" in info["target_classes"]
        assert "CRC" in info["target_classes"]

    def test_feature_names_match_species(self, sample_abundance, sample_metadata):
        """feature_names returns species names."""
        from microfactual.core.dataset import MicrobiomeDataset

        dataset = MicrobiomeDataset(
            abundance=sample_abundance,
            metadata=sample_metadata,
            target_column="disease",
        )

        assert "Bacteroides" in dataset.feature_names
        assert "Firmicutes" in dataset.feature_names


# === Extensibility Tests ===


class TestExtensibility:
    """Test that the design supports future extensions."""

    def test_from_files_accepts_custom_loader(
        self, tmp_path, sample_abundance, sample_metadata
    ):
        """from_files can be extended for different formats (BIOM, etc)."""
        # This test documents the expected extension point
        # The loader parameter should make adding BIOM support easy
        pass  # Placeholder for future BIOM support

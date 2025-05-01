import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch
from microbiome_ml.main import load_data, filter_data, clr_transform, train_model, main

# Mock data for testing
@pytest.fixture
def mock_abundance():
    data = {
        'Sample1': [0.1, 0.0, 0.0, 0.5],
        'Sample2': [0.2, 0.0, 0.0, 0.6],
        'Sample3': [0.3, 0.0, 0.0, 0.7],
        'Sample4': [0.4, 0.0, 0.0, 0.8]
    }
    df = pd.DataFrame(data, index=['Species1', 'Species2', 'Species3', 'Species4'])
    df.index.name = 'Species'
    return df

@pytest.fixture
def mock_metadata():
    data = {
        'Sample ID': ['Sample1', 'Sample2', 'Sample3', 'Sample4'],
        'Group': ['A', 'B', 'A', 'B'],
    }
    return pd.DataFrame(data)

# Unit tests
def test_load_data(mock_abundance, mock_metadata, tmp_path):
    abundance_path = tmp_path / "abundance.txt"
    metadata_path = tmp_path / "metadata.txt"
    mock_abundance.to_csv(abundance_path, sep='\t')
    mock_metadata.to_csv(metadata_path, sep='\t', index=False)

    abundance, labels = load_data(abundance_path, metadata_path, target_column='Group')
    assert abundance.shape == (4, 4)
    assert len(labels) == 4


def test_filter_data(mock_abundance):
    filtered = filter_data(mock_abundance, abundance_cutoff=0.1, prevalence_cutoff=0.5)
    assert filtered.shape[0] == 2  # Only 2 species pass the filters


def test_clr_transform(mock_abundance):
    transformed = clr_transform(mock_abundance)
    assert transformed.shape == mock_abundance.T.shape
    assert not transformed.isnull().values.any()


def test_train_model(mock_abundance, mock_metadata):
    labels = mock_metadata['Group'].astype('category').cat.codes
    clr_data = clr_transform(mock_abundance)
    model = train_model(clr_data, labels)
    assert isinstance(model.best_estimator_, RandomForestClassifier)

# Integration test
def test_main_pipeline(tmp_path, mock_abundance, mock_metadata):
    abundance_path = tmp_path / "abundance.txt"
    metadata_path = tmp_path / "metadata.txt"
    output_dir = tmp_path / "output"

    mock_abundance.to_csv(abundance_path, sep='\t')
    mock_metadata.to_csv(metadata_path, sep='\t', index=False)

    # main(str(abundance_path), str(metadata_path), target_column='Group', output_dir=str(output_dir))
    # Mock command-line arguments
    mock_args = [
        "--abundance", str(abundance_path),
        "--metadata", str(metadata_path),
        "--target", "Group",
        "--output_dir", str(output_dir)
    ]

    with patch("sys.argv", ["microbiome-ml"] + mock_args):
        main()

    # Check output files
    probabilities_file = output_dir / "predicted_probabilities.csv"
    roc_curve_file = output_dir / "roc_curve.png"

    assert probabilities_file.exists()
    assert roc_curve_file.exists()

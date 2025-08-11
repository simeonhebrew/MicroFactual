import pytest

from microfactual.main import load_data, filter_data, clr_transform


def test_load_data(mock_abundance, mock_metadata, tmp_path):
    abundance_path = tmp_path / "abundance.txt"
    metadata_path = tmp_path / "metadata.txt"
    mock_abundance.to_csv(abundance_path, sep="\t")
    mock_metadata.to_csv(metadata_path, sep="\t", index=False)

    abundance, labels = load_data(
        abundance_path, metadata_path, target_column="Group")
    assert abundance.shape == (4, 4)
    assert len(labels) == 4


def test_filter_data(mock_abundance):
    filtered = filter_data(
        mock_abundance, abundance_cutoff=0.1, prevalence_cutoff=0.5)
    assert filtered.shape[0] == 2  # Only 2 species pass the filters


def test_clr_transform(mock_abundance):
    transformed = clr_transform(mock_abundance)
    assert transformed.shape == mock_abundance.T.shape
    assert not transformed.isnull().values.any()


def test_load_data_empty_files(tmp_path):
    abundance_path = tmp_path / "empty_abundance.txt"
    metadata_path = tmp_path / "empty_metadata.txt"
    abundance_path.write_text("")
    metadata_path.write_text("")
    from microfactual.data_processing import load_data

    with pytest.raises(Exception):
        load_data(str(abundance_path), str(metadata_path))


def test_load_data_missing_column(tmp_path, mock_abundance, mock_metadata):
    abundance_path = tmp_path / "abundance.txt"
    metadata_path = tmp_path / "metadata.txt"
    mock_abundance.to_csv(abundance_path, sep="\t")
    # Remove 'Group' column
    bad_metadata = mock_metadata.drop(columns=["Group"])
    bad_metadata.to_csv(metadata_path, sep="\t", index=False)
    from microfactual.data_processing import load_data

    with pytest.raises(Exception):
        load_data(str(abundance_path), str(
            metadata_path), target_column="Group")


def test_load_data_file_not_found():
    from microfactual.data_processing import load_data

    with pytest.raises(FileNotFoundError):
        load_data("not_a_file.txt", "not_a_file2.txt")


def test_load_data_wrong_delimiter(tmp_path, mock_abundance, mock_metadata):
    abundance_path = tmp_path / "abundance.csv"
    metadata_path = tmp_path / "metadata.csv"
    mock_abundance.to_csv(abundance_path, sep=",")
    mock_metadata.to_csv(metadata_path, sep=",", index=False)
    from microfactual.data_processing import load_data

    with pytest.raises(Exception):
        load_data(str(abundance_path), str(metadata_path))


def test_filter_data_invalid_cutoffs(mock_abundance):
    from microfactual.data_processing import filter_data

    with pytest.raises(Exception):
        filter_data(mock_abundance, abundance_cutoff=-1)
    with pytest.raises(Exception):
        filter_data(mock_abundance, prevalence_cutoff=-0.1)


def test_load_data_invalid_target_column(tmp_path, mock_abundance, mock_metadata):
    abundance_path = tmp_path / "abundance.txt"
    metadata_path = tmp_path / "metadata.txt"
    mock_abundance.to_csv(abundance_path, sep="\t")
    mock_metadata.to_csv(metadata_path, sep="\t", index=False)
    from microfactual.data_processing import load_data

    with pytest.raises(Exception):
        load_data(str(abundance_path), str(
            metadata_path), target_column="NotAColumn")

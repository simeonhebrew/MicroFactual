from unittest.mock import patch

from microfactual.main import main


# Integration test
def test_main_pipeline(tmp_path, mock_abundance, mock_metadata):
    abundance_path = tmp_path / "abundance.txt"
    metadata_path = tmp_path / "metadata.txt"
    output_dir = tmp_path / "output"

    mock_abundance.to_csv(abundance_path, sep="\t")
    mock_metadata.to_csv(metadata_path, sep="\t", index=False)

    # main(str(abundance_path), str(metadata_path), target_column='Group', output_dir=str(output_dir))
    # Mock command-line arguments
    mock_args = [
        "--abundance",
        str(abundance_path),
        "--metadata",
        str(metadata_path),
        "--target",
        "Group",
        "--output_dir",
        str(output_dir),
    ]

    with patch("sys.argv", ["microbiome-ml"] + mock_args):
        main()

    # Check output files - they should be in a UUID subdirectory
    uuid_dirs = list(output_dir.glob("*"))
    assert len(uuid_dirs) == 1, "Expected exactly one UUID directory"

    uuid_dir = uuid_dirs[0]
    probabilities_file = uuid_dir / "predicted_probabilities.csv"
    roc_curve_file = uuid_dir / "roc_curve.png"

    assert probabilities_file.exists()
    assert roc_curve_file.exists()

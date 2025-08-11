from microfactual.utils import parse_args


def test_parse_args(monkeypatch):
    test_args = [
        "microfactual",
        "--abundance",
        "abundance.txt",
        "--metadata",
        "metadata.txt",
        "--target",
        "Group",
        "--output_dir",
        "output",
        "--sample_column_name",
        "Sample ID",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    args = parse_args()
    assert args.abundance == "abundance.txt"
    assert args.metadata == "metadata.txt"
    assert args.target == "Group"
    assert args.output_dir == "output"
    assert args.sample_column_name == "Sample ID"

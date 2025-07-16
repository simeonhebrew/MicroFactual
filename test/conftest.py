import pytest
import pandas as pd


# Mock data for testing
@pytest.fixture
def mock_abundance():
    data = {
        "Sample1": [0.1, 0.0, 0.0, 0.5],
        "Sample2": [0.2, 0.0, 0.0, 0.6],
        "Sample3": [0.3, 0.0, 0.0, 0.7],
        "Sample4": [0.4, 0.0, 0.0, 0.8],
    }
    df = pd.DataFrame(data, index=["Species1", "Species2", "Species3", "Species4"])
    df.index.name = "Species"
    return df


@pytest.fixture
def mock_metadata():
    data = {
        "Sample ID": ["Sample1", "Sample2", "Sample3", "Sample4"],
        "Group": ["A", "B", "A", "B"],
    }
    return pd.DataFrame(data)

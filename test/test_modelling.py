from sklearn.ensemble import RandomForestClassifier

from microfactual.main import clr_transform, train_model


def test_train_model(mock_abundance, mock_metadata):
    labels = mock_metadata["Group"].astype("category").cat.codes
    clr_data = clr_transform(mock_abundance)
    model = train_model(clr_data, labels)
    assert isinstance(model.best_estimator_, RandomForestClassifier)

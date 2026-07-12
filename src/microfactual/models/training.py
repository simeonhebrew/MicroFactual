"""Legacy functional training helper used by the CLI pipeline.

The modern, sklearn-native path is
:class:`microfactual.models.classifiers.MicrobiomeClassifier`.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.model_selection import GridSearchCV as GridSearchCVType


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 2,
    n_jobs: int | None = None,
    n_estimators: int = 100,
) -> GridSearchCVType:
    """Train a Random Forest with cross-validated ``max_features`` tuning.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Features matrix (samples x features).
    y : np.ndarray or pd.Series
        Target vector (labels).
    cv_splits : int, optional
        Number of cross-validation splits (default: 2).
    n_jobs : int or None, optional
        Number of parallel jobs. None means 1 unless in a joblib context.
    n_estimators : int, optional
        Number of trees in the forest (default: 100).

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object with the best model.

    """
    param_grid = {"max_features": np.unique(np.linspace(1, X.shape[1], 5, dtype=int))}
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    cv = RepeatedKFold(n_splits=cv_splits, n_repeats=2, random_state=42)

    grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=n_jobs)
    grid_search.fit(X, y)
    return grid_search

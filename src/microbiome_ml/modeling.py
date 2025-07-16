"""Model training utilities for microbiome-ml.

This module provides functions for training machine learning models (e.g., Random Forest)
on microbiome datasets with cross-validation and hyperparameter tuning.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.model_selection import GridSearchCV as GridSearchCVType

# Set environment variables to control threading behavior
# and prevent multiprocessing issues
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 2,
    n_jobs: int | None = None,
    n_estimators: int = 100,
) -> GridSearchCVType:
    """Train Random Forest classifier with cross-validation.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Features matrix (samples x features).
    y : np.ndarray or pd.Series
        Target vector (labels).
    cv_splits : int, optional
        Number of cross-validation splits (default: 2).
    n_jobs : int or None, optional
        Number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. Set to 1 to avoid multiprocessing issues.
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

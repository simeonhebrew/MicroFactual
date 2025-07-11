from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, GridSearchCV
import numpy as np

def train_model(X, y, cv_splits=2, n_jobs=None, n_estimators=100):
    """Train Random Forest classifier with cross-validation
    
    Parameters:
    -----------
    X : array-like
        Features matrix
    y : array-like
        Target vector
    cv_splits : int, default=2
        Number of cross-validation splits
    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context.
        Set to 1 to avoid multiprocessing issues.
    """
    param_grid = {'max_features': np.unique(np.linspace(1, X.shape[1], 5, dtype=int))}
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    cv = RepeatedKFold(n_splits=cv_splits, n_repeats=2, random_state=42)

    grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=n_jobs)
    grid_search.fit(X, y)
    return grid_search
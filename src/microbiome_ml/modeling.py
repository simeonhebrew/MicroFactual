from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, GridSearchCV
import numpy as np

def train_model(X, y, cv_splits=2):
    """Train Random Forest classifier with cross-validation"""
    param_grid = {'max_features': np.unique(np.linspace(1, X.shape[1], 5, dtype=int))}
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = RepeatedKFold(n_splits=cv_splits, n_repeats=2, random_state=42)

    grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search
"""Microbiome-optimized classifiers with built-in preprocessing.

The MicrobiomeClassifier provides a batteries-included approach to
microbiome classification with sensible defaults for preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from microfactual.preprocessing.transforms import (
    AbundanceFilter,
    CLRTransform,
    PrevalenceFilter,
)


class MicrobiomeClassifier(BaseEstimator, ClassifierMixin):
    """Classifier with built-in microbiome preprocessing.

    This provides a simple interface for microbiome classification with
    sensible defaults. It wraps sklearn classifiers with optional
    preprocessing steps.

    Parameters
    ----------
    algorithm : str, default="random_forest"
        The classification algorithm to use.
        Options: "random_forest", "svm", "logistic"
    preprocessing : str, list, or None, default="auto"
        Preprocessing strategy:
        - "auto": Apply abundance filter, prevalence filter, CLR transform
        - None: No preprocessing (use raw features)
        - list: Custom list of sklearn-compatible transformers
    **model_params
        Additional parameters passed to the underlying classifier.

    Examples
    --------
    >>> clf = MicrobiomeClassifier()
    >>> clf.fit(X, y)
    >>> predictions = clf.predict(X_test)
    """

    ALGORITHMS = {
        "random_forest": RandomForestClassifier,
        "logistic": LogisticRegression,
    }

    def __init__(
        self,
        algorithm: str = "random_forest",
        preprocessing: str | list | None = "auto",
        **model_params,
    ):
        self.algorithm = algorithm
        self.preprocessing = preprocessing
        self.model_params = model_params

    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix (samples x features).
        y : pd.Series or np.ndarray
            Target vector.

        Returns
        -------
        self
        """
        # Build pipeline steps
        steps = []

        if self.preprocessing == "auto":
            steps.extend(
                [
                    ("abundance", AbundanceFilter(min_abundance=1e-6)),
                    ("prevalence", PrevalenceFilter(min_prevalence=0.01)),
                    ("clr", CLRTransform()),
                ]
            )
        elif isinstance(self.preprocessing, list):
            for i, step in enumerate(self.preprocessing):
                steps.append((f"step_{i}", step))
        # If None, no preprocessing

        # Add classifier
        if self.algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                f"Options: {list(self.ALGORITHMS.keys())}"
            )

        clf_class = self.ALGORITHMS[self.algorithm]

        # Set default params
        default_params = {}
        if self.algorithm == "random_forest":
            default_params = {"n_estimators": 100, "random_state": 42}
        elif self.algorithm == "logistic":
            default_params = {"random_state": 42, "max_iter": 1000}

        # Override with user params
        clf_params = {**default_params, **self.model_params}
        steps.append(("classifier", clf_class(**clf_params)))

        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(X, y)
        self.classes_ = self.pipeline_.classes_

        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix (samples x features).

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels.
        """
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix (samples x features).

        Returns
        -------
        proba : np.ndarray
            Probability for each class, shape (n_samples, n_classes).
        """
        return self.pipeline_.predict_proba(X)

    @property
    def feature_importances_(self):
        """Get feature importances if available."""
        clf = self.pipeline_.named_steps.get("classifier")
        if hasattr(clf, "feature_importances_"):
            return clf.feature_importances_
        return None

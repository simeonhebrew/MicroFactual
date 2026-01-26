"""Sklearn-compatible preprocessing transforms for microbiome data.

All transforms follow sklearn's fit/transform pattern and can be used
in sklearn.pipeline.Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AbundanceFilter(BaseEstimator, TransformerMixin):
    """Filter features by minimum mean abundance.

    Parameters
    ----------
    min_abundance : float, default=1e-6
        Minimum mean abundance required to retain a feature.

    Examples
    --------
    >>> filt = AbundanceFilter(min_abundance=0.01)
    >>> X_filtered = filt.fit_transform(X)

    """

    def __init__(self, min_abundance: float = 1e-6):
        self.min_abundance = min_abundance

    def fit(self, X: pd.DataFrame, y=None):
        """Learn which features pass the abundance threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Abundance data (samples x features).
        y : ignored

        """
        if isinstance(X, pd.DataFrame):
            self.mask_ = X.mean(axis=0) >= self.min_abundance
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.mask_ = np.mean(X, axis=0) >= self.min_abundance
            self.feature_names_in_ = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the abundance filter.

        Parameters
        ----------
        X : pd.DataFrame
            Abundance data (samples x features).

        Returns
        -------
        pd.DataFrame
            Filtered data with low-abundance features removed.

        """
        if isinstance(X, pd.DataFrame):
            # Use values to avoid alignment issues if columns were renamed
            mask = (
                self.mask_.values if isinstance(self.mask_, pd.Series) else self.mask_
            )
            return X.loc[:, mask]
        return X[:, self.mask_]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        array of str
            Transformed feature names.

        """
        if input_features is None:
            input_features = self.feature_names_in_
        if input_features is None:
            raise ValueError(
                "input_features must be specified if not fitted on DataFrame"
            )

        input_features = np.array(input_features, dtype=object)
        return input_features[self.mask_]


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """Filter features by minimum prevalence (fraction of non-zero samples).

    Parameters
    ----------
    min_prevalence : float, default=0.05
        Minimum fraction of samples in which a feature must be present.
        Must be between 0 and 1.

    Examples
    --------
    >>> filt = PrevalenceFilter(min_prevalence=0.1)
    >>> X_filtered = filt.fit_transform(X)

    """

    def __init__(self, min_prevalence: float = 0.05):
        self.min_prevalence = min_prevalence

    def fit(self, X: pd.DataFrame, y=None):
        """Learn which features pass the prevalence threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Abundance data (samples x features).
        y : ignored

        """
        if isinstance(X, pd.DataFrame):
            self.mask_ = (X > 0).mean(axis=0) >= self.min_prevalence
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.mask_ = np.mean(X > 0, axis=0) >= self.min_prevalence
            self.feature_names_in_ = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the prevalence filter.

        Parameters
        ----------
        X : pd.DataFrame
            Abundance data (samples x features).

        Returns
        -------
        pd.DataFrame
            Filtered data with rare features removed.

        """
        if isinstance(X, pd.DataFrame):
            # Use values to avoid alignment issues if columns were renamed
            mask = (
                self.mask_.values if isinstance(self.mask_, pd.Series) else self.mask_
            )
            return X.loc[:, mask]
        return X[:, self.mask_]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        array of str
            Transformed feature names.

        """
        if input_features is None:
            input_features = self.feature_names_in_
        if input_features is None:
            raise ValueError(
                "input_features must be specified if not fitted on DataFrame"
            )

        input_features = np.array(input_features, dtype=object)
        return input_features[self.mask_]


class CLRTransform(BaseEstimator, TransformerMixin):
    """Centered Log-Ratio (CLR) transformation for compositional data.

    The CLR transformation is appropriate for microbiome abundance data
    which is compositional (relative abundances sum to 1).

    Parameters
    ----------
    pseudocount : float, default=1e-6
        Small value added to zeros before log transformation.

    Examples
    --------
    >>> clr = CLRTransform()
    >>> X_transformed = clr.fit_transform(X)

    """

    def __init__(self, pseudocount: float = 1e-6):
        self.pseudocount = pseudocount

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer.

        CLR is a stateless transformation, so this just validates input.

        Parameters
        ----------
        X : pd.DataFrame
            Abundance data (samples x features).
        y : ignored

        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the CLR transformation.

        Parameters
        ----------
        X : pd.DataFrame
            Abundance data (samples x features).

        Returns
        -------
        pd.DataFrame
            CLR-transformed data with the same shape.

        """
        if isinstance(X, pd.DataFrame):
            X_pseudo = X + self.pseudocount
            log_X = np.log(X_pseudo)
            geometric_mean = log_X.mean(axis=1)
            return log_X.sub(geometric_mean, axis=0)
        else:
            X_pseudo = X + self.pseudocount
            log_X = np.log(X_pseudo)
            geometric_mean = np.mean(log_X, axis=1, keepdims=True)
            return log_X - geometric_mean

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        array of str
            Transformed feature names.

        """
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                input_features = self.feature_names_in_
            else:
                # Should we support non-dataframe default?
                # If no info, we can't really guess safely without assuming 1-1 mapping
                pass

        if input_features is None:
            # Standard sklearn behavior usually requires input features or fitted names
            # Since CLR is 1-to-1, we can just return input_features if provided.
            # If not provided and no feature_names_in_, we might fail or return None.
            # Let's assume input_features is provided or we throw.
            raise ValueError(
                "input_features must be specified if not fitted on DataFrame"
            )

        return np.array(input_features, dtype=object)

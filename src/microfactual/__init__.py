"""Microfactual: Machine Learning for Microbiome Data Analysis."""

import warnings
from typing import Any

from sklearn.model_selection import cross_validate

# Core data structures and interfaces
from .core import BaseModel, MicrobiomeDataset

# Keep existing functions available for backward compatibility
from .data_processing import clr_transform, filter_data, load_data

# New explainability module
from .explainability import BaseExplainer, DiCEExplainer
from .main import main
from .main import run_pipeline as _run_pipeline_internal
from .modeling import train_model

# New architecture components
from .models.classifiers import MicrobiomeClassifier
from .preprocessing.transforms import AbundanceFilter, CLRTransform, PrevalenceFilter
from .utils import get_logger, parse_args, save_results
from .visualisation import save_roc_curve

# New visualization module
from .visualization import (
    launch_dashboard,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc,
)


def run_pipeline(*args, **kwargs):
    """Deprecated: Use mf.classify() instead."""
    warnings.warn(
        "run_pipeline is deprecated and will be removed in v1.0. "
        "Use microfactual.classify() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _run_pipeline_internal(*args, **kwargs)


def classify(
    abundance_file: str,
    metadata_file: str,
    target_column: str,
    *,
    sample_column: str = "Sample ID",
    algorithm: str = "random_forest",
    cv_folds: int = 5,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """One-function classification pipeline for microbiome data.

    Parameters
    ----------
    abundance_file : str
        Path to abundance data file (tab-separated, species x samples).
    metadata_file : str
        Path to metadata file (tab-separated).
    target_column : str
        Column name for the target variable.
    sample_column : str, default="Sample ID"
        Column name for sample IDs in metadata.
    algorithm : str, default="random_forest"
        Classification algorithm ("random_forest", "logistic").
    cv_folds : int, default=5
        Number of cross-validation folds.
    output_dir : str, optional
        Directory to save results.

    Returns
    -------
    dict
        Results containing:
        - "model": Fitted MicrobiomeClassifier
        - "dataset": MicrobiomeDataset
        - "cv_scores": Cross-validation scores

    Examples
    --------
    >>> import microfactual as mf
    >>> results = mf.classify(
    ...     "data/abundance.tsv", "data/metadata.tsv", target_column="disease"
    ... )
    >>> print(f"CV Accuracy: {results['cv_scores']['test_accuracy']:.3f}")

    """
    # Load data using MicrobiomeDataset
    dataset = MicrobiomeDataset.from_files(
        abundance_file=abundance_file,
        metadata_file=metadata_file,
        target_column=target_column,
        sample_column=sample_column,
    )

    # Create classifier
    model = MicrobiomeClassifier(algorithm=algorithm, preprocessing="auto")

    # Cross-validate
    cv_results = cross_validate(
        model,
        dataset.X,
        dataset.y,
        cv=cv_folds,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=True,
    )

    # Fit final model on all data
    model.fit(dataset.X, dataset.y)

    # Save if output_dir specified
    if output_dir:
        save_results(
            output_dir,
            dataset.X,
            model.predict_proba(dataset.X)[:, 1],
            dataset.y,
            get_logger(__name__),
        )
        save_roc_curve(
            output_dir,
            dataset.y,
            model.predict_proba(dataset.X)[:, 1],
            get_logger(__name__),
        )

    return {
        "model": model,
        "dataset": dataset,
        "cv_scores": {k: v.mean() for k, v in cv_results.items()},
    }


# Version information
__version__ = "0.2.0"
__author__ = "Lawrence Adu-Gyamfi, Simeon Hebrew"

# Main API exports
__all__ = [
    # High-level API
    "classify",
    "MicrobiomeClassifier",
    # Core classes
    "MicrobiomeDataset",
    "BaseModel",
    # Preprocessing
    "AbundanceFilter",
    "PrevalenceFilter",
    "CLRTransform",
    # Legacy (deprecated)
    "run_pipeline",
    "main",
    # Data processing
    "load_data",
    "filter_data",
    "clr_transform",
    # Modeling
    "train_model",
    # Visualization
    "plot_roc",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "launch_dashboard",
    "save_roc_curve",
    # Explainability
    "BaseExplainer",
    "DiCEExplainer",
    # Utilities
    "get_logger",
    "parse_args",
    "save_results",
    # Package info
    "__version__",
    "__author__",
]

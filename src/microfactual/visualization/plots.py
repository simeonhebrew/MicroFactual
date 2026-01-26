"""Plotting functions for microbiome ML results.

All plot functions return matplotlib Figure objects for flexibility.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def plot_roc(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    title: str = "ROC Curve",
) -> plt.Figure:
    """Plot ROC curve with AUC score.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    title : str, default="ROC Curve"
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> fig = plot_roc(y_test, model.predict_proba(X_test)[:, 1])
    >>> fig.savefig("roc.png")

    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return fig


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: Sequence[str] | None = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    labels : list of str, optional
        Display labels for classes.
    title : str, default="Confusion Matrix"
        Plot title.
    cmap : str, default="Blues"
        Colormap for the matrix.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> fig = plot_confusion_matrix(y_test, y_pred, labels=["Healthy", "Disease"])

    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=cmap, colorbar=True)
    ax.set_title(title)

    return fig


def plot_feature_importance(
    importance: pd.Series | np.ndarray | object,
    feature_names: Sequence[str] | None = None,
    top_n: int | None = None,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Plot feature importance as horizontal bar chart.

    Parameters
    ----------
    importance : pd.Series, np.ndarray, or model with feature_importances_
        Feature importance scores. Can be:
        - pd.Series with feature names as index
        - np.ndarray (requires feature_names)
        - Model object with feature_importances_ attribute
    feature_names : list of str, optional
        Feature names (required if importance is array or model).
    top_n : int, optional
        Show only top N most important features.
    title : str, default="Feature Importance"
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> fig = plot_feature_importance(model.feature_importances_, feature_names)
    >>> fig = plot_feature_importance(importance_series, top_n=20)

    """
    # Handle different input types
    if hasattr(importance, "feature_importances_"):
        # It's a model
        imp_values = importance.feature_importances_
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(imp_values))]
        importance = pd.Series(imp_values, index=feature_names)
    elif isinstance(importance, np.ndarray):
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importance))]
        importance = pd.Series(importance, index=feature_names)
    elif not isinstance(importance, pd.Series):
        raise TypeError(
            f"importance must be pd.Series, np.ndarray, or model with "
            f"feature_importances_. Got {type(importance)}"
        )

    # Sort and optionally limit
    importance = importance.sort_values(ascending=True)
    if top_n is not None:
        importance = importance.tail(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance) * 0.3)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance)))
    ax.barh(range(len(importance)), importance.values, color=colors)
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(importance.index)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    return fig

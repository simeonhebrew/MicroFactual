"""Visualization utilities for microbiome-ml."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from microbiome_ml.utils import get_logger

app_logger = get_logger(__name__)


def plot_roc(
    y_true: pd.Series,
    y_probs: pd.Series,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot ROC curve and optionally save or display it.

    Parameters
    ----------
    y_true : pd.Series
        True binary labels.
    y_probs : pd.Series
        Predicted probabilities for the positive class.
    save_path : str or None, optional
        Directory to save the ROC curve plot (default: None).
    show : bool, optional
        Whether to display the plot interactively (default: False).

    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend()
    if save_path:
        save_path = os.path.join(save_path, "roc_curve.png")
        plt.savefig(save_path)
        app_logger.info(f"ROC curve saved to {save_path}")
    if show:
        plt.show()


def save_roc_curve(
    output_dir: str,
    labels: pd.Series,
    probs: pd.Series,
    logger: logging.Logger | None = None,
) -> None:
    """Plot and save the ROC curve to the output directory.

    Parameters
    ----------
    output_dir : str
        Directory to save the ROC curve plot.
    labels : pd.Series
        True binary labels.
    probs : pd.Series
        Predicted probabilities for the positive class.
    logger : logging.Logger, optional
        Logger for info messages (default: module logger).

    """
    if logger is None:
        logger = get_logger(__name__)
    plot_roc(labels, probs, save_path=output_dir, show=False)
    logger.info(f"ROC curve saved to {os.path.join(output_dir, 'roc_curve.png')}")

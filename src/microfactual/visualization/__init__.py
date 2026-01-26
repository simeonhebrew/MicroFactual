"""Visualization module for microfactual."""

from microfactual.visualization.dashboard import launch_dashboard
from microfactual.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc,
)

__all__ = [
    "plot_roc",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "launch_dashboard",
]

"""Visualization module for microfactual."""

from microfactual.visualization.counterfactual_plots import (
    plot_counterfactual_heatmap,
)
from microfactual.visualization.exploration import (
    explore,
    plot_abundance_histogram,
    plot_prevalence_abundance,
    plot_prevalence_histogram,
)
from microfactual.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc,
)

__all__ = [
    "plot_roc",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "explore",
    "plot_abundance_histogram",
    "plot_prevalence_histogram",
    "plot_prevalence_abundance",
    "plot_counterfactual_heatmap",
]

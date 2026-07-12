"""Data-exploration plots for choosing preprocessing cutoffs.

These helpers turn MicroFactual's filtering defaults (``AbundanceFilter``,
``PrevalenceFilter``) into data-driven decisions: plot how per-taxon abundance
and prevalence are distributed, overlay the proposed cutoffs, and report how
many taxa survive them.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Small floor so log10 of the abundance cutoff is well-defined for annotation.
_LOG_FLOOR = 1e-12


def _get_abundance(data: Any) -> pd.DataFrame:
    """Return a features x samples abundance frame from a dataset or frame.

    Parameters
    ----------
    data : MicrobiomeDataset or pd.DataFrame
        Either a ``MicrobiomeDataset`` (its ``.abundance`` is used) or a
        features x samples abundance DataFrame.

    Returns
    -------
    pd.DataFrame
        Abundance matrix with features as rows and samples as columns.

    """
    abundance = getattr(data, "abundance", data)
    if not isinstance(abundance, pd.DataFrame):
        raise TypeError(
            "Expected a MicrobiomeDataset or a features x samples DataFrame, "
            f"got {type(data).__name__}."
        )
    return abundance


def _per_taxon_stats(abundance: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-taxon mean abundance and prevalence.

    Parameters
    ----------
    abundance : pd.DataFrame
        Abundance matrix (features x samples).

    Returns
    -------
    tuple of np.ndarray
        ``(mean_abundance, prevalence)``, one value per taxon. Prevalence is the
        fraction of samples in which the taxon is present (> 0).

    """
    values = abundance.to_numpy()
    mean_abundance = values.mean(axis=1)
    prevalence = (values > 0).mean(axis=1)
    return mean_abundance, prevalence


def plot_abundance_histogram(
    data: Any,
    abundance_cutoff: float | None = 1e-6,
    ax: plt.Axes | None = None,
    bins: int = 50,
) -> plt.Axes:
    """Histogram of per-taxon mean abundance on a log10 axis.

    Microbiome abundances span many orders of magnitude and the distribution is
    often bimodal; the trough is a natural place to set the abundance cutoff.

    Parameters
    ----------
    data : MicrobiomeDataset or pd.DataFrame
        Data to summarise (see :func:`_get_abundance`).
    abundance_cutoff : float, optional
        If given, draw a vertical line at this abundance and annotate how many
        taxa fall at or above it.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure/axes is created if omitted.
    bins : int, default=50
        Number of histogram bins.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    """
    abundance = _get_abundance(data)
    mean_abundance, _ = _per_taxon_stats(abundance)

    positive = mean_abundance[mean_abundance > 0]
    n_all_zero = int((mean_abundance <= 0).sum())

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    log_mean = np.log10(positive)
    ax.hist(log_mean, bins=bins, color="#4C72B0", alpha=0.8)
    ax.set_xlabel("log10(mean relative abundance)")
    ax.set_ylabel("Number of taxa")
    ax.set_title("Per-taxon mean abundance")

    if abundance_cutoff is not None:
        line_x = np.log10(max(abundance_cutoff, _LOG_FLOOR))
        n_retained = int((mean_abundance >= abundance_cutoff).sum())
        ax.axvline(
            line_x,
            color="#C44E52",
            linestyle="--",
            label=f"cutoff={abundance_cutoff:g}\n{n_retained} taxa ≥ cutoff",
        )
        ax.legend(fontsize=8, loc="upper right")

    if n_all_zero:
        ax.text(
            0.02,
            0.98,
            f"{n_all_zero} all-zero taxa omitted",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            color="gray",
        )
    return ax


def plot_prevalence_histogram(
    data: Any,
    prevalence_cutoff: float | None = 0.01,
    ax: plt.Axes | None = None,
    bins: int = 50,
) -> plt.Axes:
    """Histogram of per-taxon prevalence (fraction of samples present).

    Prevalence distributions are typically U-shaped (taxa present in almost no
    samples or almost all). The plot helps place the prevalence cutoff.

    Parameters
    ----------
    data : MicrobiomeDataset or pd.DataFrame
        Data to summarise.
    prevalence_cutoff : float, optional
        If given, draw a vertical line at this prevalence and annotate how many
        taxa fall at or above it.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure/axes is created if omitted.
    bins : int, default=50
        Number of histogram bins.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    """
    abundance = _get_abundance(data)
    _, prevalence = _per_taxon_stats(abundance)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.hist(prevalence, bins=bins, color="#55A868", alpha=0.8)
    ax.set_xlabel("Prevalence (fraction of samples present)")
    ax.set_ylabel("Number of taxa")
    ax.set_title("Per-taxon prevalence")

    if prevalence_cutoff is not None:
        n_retained = int((prevalence >= prevalence_cutoff).sum())
        ax.axvline(
            prevalence_cutoff,
            color="#C44E52",
            linestyle="--",
            label=f"cutoff={prevalence_cutoff:g}\n{n_retained} taxa ≥ cutoff",
        )
        ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_prevalence_abundance(
    data: Any,
    abundance_cutoff: float | None = 1e-6,
    prevalence_cutoff: float | None = 0.01,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Joint scatter of per-taxon prevalence vs. mean abundance.

    Each point is one taxon. With both cutoffs supplied, the retained region is
    shaded and the number of surviving taxa is annotated — the clearest single
    view for choosing both filters together.

    Parameters
    ----------
    data : MicrobiomeDataset or pd.DataFrame
        Data to summarise.
    abundance_cutoff : float, optional
        Abundance threshold; drawn as a vertical line.
    prevalence_cutoff : float, optional
        Prevalence threshold; drawn as a horizontal line.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure/axes is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    """
    abundance = _get_abundance(data)
    mean_abundance, prevalence = _per_taxon_stats(abundance)

    positive = mean_abundance > 0
    log_mean = np.log10(np.where(positive, mean_abundance, _LOG_FLOOR))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        log_mean[positive],
        prevalence[positive],
        s=8,
        alpha=0.4,
        color="#4C72B0",
        edgecolors="none",
    )
    ax.set_xlabel("log10(mean relative abundance)")
    ax.set_ylabel("Prevalence (fraction of samples)")
    ax.set_title("Prevalence vs. abundance")

    retained = np.ones_like(mean_abundance, dtype=bool)
    if abundance_cutoff is not None:
        ax.axvline(
            np.log10(max(abundance_cutoff, _LOG_FLOOR)), color="#C44E52", linestyle="--"
        )
        retained &= mean_abundance >= abundance_cutoff
    if prevalence_cutoff is not None:
        ax.axhline(prevalence_cutoff, color="#C44E52", linestyle="--")
        retained &= prevalence >= prevalence_cutoff

    if abundance_cutoff is not None or prevalence_cutoff is not None:
        n_ret = int(retained.sum())
        ax.text(
            0.98,
            0.02,
            f"{n_ret} / {len(mean_abundance)} taxa retained",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round", "fc": "white", "ec": "#C44E52", "alpha": 0.8},
        )
    return ax


def explore(
    data: Any,
    abundance_cutoff: float | None = 1e-6,
    prevalence_cutoff: float | None = 0.01,
    figsize: tuple[float, float] = (16, 4),
) -> plt.Figure:
    """Render a cutoff-diagnostics panel for a microbiome dataset.

    Combines the abundance histogram, prevalence histogram, and the joint
    prevalence-vs-abundance scatter into one figure, all sharing the proposed
    ``AbundanceFilter`` / ``PrevalenceFilter`` cutoffs, so you can eyeball
    whether the defaults suit your data before fitting a model.

    Parameters
    ----------
    data : MicrobiomeDataset or pd.DataFrame
        Data to summarise (a ``MicrobiomeDataset`` or a features x samples
        abundance DataFrame).
    abundance_cutoff : float, optional
        Proposed abundance cutoff (matches ``AbundanceFilter(min_abundance=...)``).
    prevalence_cutoff : float, optional
        Proposed prevalence cutoff (matches ``PrevalenceFilter(min_prevalence=...)``).
    figsize : tuple of float, default=(16, 4)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled 1x3 panel figure.

    """
    abundance = _get_abundance(data)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_abundance_histogram(abundance, abundance_cutoff=abundance_cutoff, ax=axes[0])
    plot_prevalence_histogram(
        abundance, prevalence_cutoff=prevalence_cutoff, ax=axes[1]
    )
    plot_prevalence_abundance(
        abundance,
        abundance_cutoff=abundance_cutoff,
        prevalence_cutoff=prevalence_cutoff,
        ax=axes[2],
    )

    n_features, n_samples = abundance.shape
    fig.suptitle(
        f"Cutoff diagnostics — {n_features} taxa × {n_samples} samples",
        fontsize=12,
    )
    fig.tight_layout()
    return fig

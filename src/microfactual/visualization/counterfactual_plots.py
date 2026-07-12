"""Visualizations for counterfactual explanations."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from microfactual.explainability.plausibility import counterfactual_concordance


def _class_label(result: Any, cls: Any) -> str:
    """Map an integer class code to its name via ``result.class_names`` if set."""
    names = getattr(result, "class_names", None)
    if (
        names is not None
        and isinstance(cls, (int, np.integer))
        and 0 <= cls < len(names)
    ):
        return str(names[cls])
    return str(cls)


def plot_counterfactual_heatmap(
    result: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    reference_class: Any,
    comparison_class: Any = None,
    cf_index: int = 0,
    top_n: int | None = 15,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of a counterfactual against class-reference distributions.

    For the taxa a counterfactual changes, shows the sample's original value, the
    counterfactual value, and the class medians — all expressed in
    **reference-class standard deviations** (0 = reference-class centre). This
    makes it easy to see whether the counterfactual moves the sample *toward* the
    reference (e.g. control) distribution, and whether it overshoots into
    implausible territory. The title reports the concordance-with-reference score.

    Parameters
    ----------
    result : CounterfactualResult
        A result from
        :func:`~microfactual.explainability.counterfactuals.explain_counterfactual`.
    X : pd.DataFrame
        Feature matrix (samples x features) used to compute class references.
    y : array-like
        Class labels aligned with ``X``.
    reference_class : Any
        Label value in ``y`` to treat as the reference (e.g. the control/healthy
        class the counterfactual should move toward).
    comparison_class : Any, optional
        A second class whose median is shown for contrast (e.g. the disease
        class). If None, only the reference median is shown.
    cf_index : int, default=0
        Which counterfactual (row) of ``result`` to display.
    top_n : int, optional
        Show only the ``top_n`` taxa with the largest absolute change. None shows
        all changed taxa.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure/axes is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.

    """
    changes = result.changes(cf_index)
    if len(changes) == 0:
        raise ValueError("The selected counterfactual changes no features.")

    changes = changes.reindex(changes["delta"].abs().sort_values(ascending=False).index)
    if top_n is not None:
        changes = changes.head(top_n)
    taxa = list(changes["feature"])

    y_arr = np.asarray(y)
    ref_mask = y_arr == reference_class
    if not ref_mask.any():
        raise ValueError(f"No samples with reference_class={reference_class!r} in y.")
    reference = X.loc[ref_mask, taxa]
    mu = reference.mean()
    sd = reference.std().replace(0, 1)

    def z(values: pd.Series) -> pd.Series:
        return (values - mu) / sd

    ref_label = _class_label(result, reference_class)
    columns = {
        "Original": z(result.original[taxa]),
        "Counterfactual": z(result.counterfactuals.iloc[cf_index][taxa]),
        f"{ref_label} median": z(reference.median()),
    }
    if comparison_class is not None:
        comp_mask = y_arr == comparison_class
        comp_label = _class_label(result, comparison_class)
        columns[f"{comp_label} median"] = z(X.loc[comp_mask, taxa].median())

    matrix = pd.DataFrame(columns, index=taxa)

    concordance = counterfactual_concordance(result, X, y, reference_class)

    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 * matrix.shape[1] + 3, 0.45 * len(taxa) + 2))

    im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(taxa)))
    ax.set_yticklabels([t[:40] for t in taxa], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix.iat[i, j]:.1f}", ha="center", va="center", fontsize=7
            )
    ax.set_title(
        f"Counterfactual vs class references ({ref_label}-SD units)\n"
        f"concordance with {ref_label} = {concordance:.0%}",
        fontsize=10,
    )
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label(f"{ref_label}-SDs (0 = {ref_label} centre)")
    return ax

"""Reference-class plausibility helpers for counterfactuals.

Vanilla DiCE can propose counterfactual values far outside anything seen in the
data. These helpers (1) derive a plausible value range from a reference class to
constrain the search (``permitted_range``), and (2) score how well a
counterfactual moves a sample *toward* that reference class.
"""

from typing import Any

import numpy as np
import pandas as pd


def plausible_range(
    X: pd.DataFrame,
    y: Any,
    reference_class: Any,
    *,
    q: tuple[float, float] = (0.05, 0.95),
    features: list[str] | None = None,
) -> dict[str, list[float]]:
    """Per-feature ``[low, high]`` bounds from a reference class's quantiles.

    Feed the result to ``explain_counterfactual(..., permitted_range=...)`` so
    counterfactuals are confined to values actually observed in the reference
    (e.g. control) class, rather than extrapolating to implausible extremes.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features), in the model's input space.
    y : array-like
        Class labels aligned with ``X``.
    reference_class : Any
        The label value in ``y`` whose distribution defines "plausible".
    q : tuple of float, default=(0.05, 0.95)
        Lower/upper quantiles used as the bounds.
    features : list of str, optional
        Restrict to these features; defaults to all columns of ``X``.

    Returns
    -------
    dict
        ``{feature: [low, high]}`` suitable for DiCE's ``permitted_range``.

    """
    mask = np.asarray(y) == reference_class
    if not mask.any():
        raise ValueError(f"No samples with reference_class={reference_class!r} in y.")
    cols = list(X.columns) if features is None else list(features)
    reference = X.loc[mask, cols]
    lo = reference.quantile(q[0])
    hi = reference.quantile(q[1])
    return {c: [float(lo[c]), float(hi[c])] for c in cols}


def counterfactual_concordance(
    result: Any,
    X: pd.DataFrame,
    y: Any,
    reference_class: Any,
) -> float:
    """Fraction of changed taxa whose counterfactual moves toward the reference.

    For each changed taxon, checks whether the counterfactual's change is in the
    same direction as the gap between the sample's original value and the
    reference-class median. A high score means the counterfactual shifts the
    sample *toward* the reference distribution rather than in some arbitrary
    model-exploiting direction — a cheap plausibility signal.

    Parameters
    ----------
    result : CounterfactualResult
        A result from :func:`~microfactual.explainability.counterfactuals.explain_counterfactual`.
    X : pd.DataFrame
        Feature matrix used to compute the reference median.
    y : array-like
        Class labels aligned with ``X``.
    reference_class : Any
        The label value in ``y`` to move toward.

    Returns
    -------
    float
        Concordance in ``[0, 1]``, or ``nan`` if there are no changes.

    """
    changes = result.changes()
    if len(changes) == 0:
        return float("nan")
    mask = np.asarray(y) == reference_class
    if not mask.any():
        raise ValueError(f"No samples with reference_class={reference_class!r} in y.")
    reference_median = X.loc[mask].median()

    toward = [
        np.sign(row["delta"])
        == np.sign(reference_median[row["feature"]] - row["original"])
        for _, row in changes.iterrows()
    ]
    return float(np.mean(toward))

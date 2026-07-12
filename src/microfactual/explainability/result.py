"""Structured, interpretable counterfactual results.

Wraps the raw DiCE output into an object that answers the questions a user
actually has: *which* taxa changed, by how much, in which direction, and does
the counterfactual really flip the model's prediction?
"""

from typing import Any

import numpy as np
import pandas as pd

# Values closer than this are treated as "unchanged".
_ATOL = 1e-8


def sparsify_counterfactual(
    original: np.ndarray,
    counterfactual: np.ndarray,
    model: Any,
    atol: float = _ATOL,
) -> np.ndarray:
    """Greedily revert changed features while preserving the flipped prediction.

    DiCE (especially the genetic method on continuous features) often perturbs
    *every* feature. This reduces a counterfactual to a near-minimal set of
    changes: changed features are reverted to their original value one at a time,
    smallest change first, keeping a revert only when the model's prediction is
    unchanged. The result is a sparse, actionable counterfactual.

    Parameters
    ----------
    original : np.ndarray
        The original sample's feature values (1D).
    counterfactual : np.ndarray
        The counterfactual feature values (1D, same order as ``original``).
    model : Any
        Fitted classifier exposing ``predict`` on a 2D array.
    atol : float, default=1e-8
        Absolute tolerance below which a feature is considered unchanged.

    Returns
    -------
    np.ndarray
        A sparsified counterfactual with the same predicted class as the input
        counterfactual.

    """
    original = np.asarray(original, dtype=float)
    work = np.asarray(counterfactual, dtype=float).copy()

    target_pred = int(model.predict(work.reshape(1, -1))[0])

    changed = np.flatnonzero(np.abs(work - original) > atol)
    # Revert smallest-magnitude changes first.
    order = sorted(changed, key=lambda j: abs(work[j] - original[j]))
    for j in order:
        trial = work.copy()
        trial[j] = original[j]
        if int(model.predict(trial.reshape(1, -1))[0]) == target_pred:
            work = trial
    return work


class CounterfactualResult:
    """Interpretable wrapper around a set of counterfactuals for one sample.

    Parameters
    ----------
    original : array-like
        The query sample's feature values.
    counterfactuals : array-like
        Counterfactual rows (n_cfs x n_features), already sparsified if desired.
    model : Any
        Fitted classifier used to (re)validate predictions.
    feature_names : list of str
        Names of the features, in column order.
    class_names : list of str, optional
        Human-readable class labels, indexed by predicted integer class.
    raw : Any, optional
        The underlying DiCE explanation object, kept for advanced use.
    atol : float, default=1e-8
        Tolerance below which a feature is considered unchanged.

    """

    def __init__(
        self,
        original: Any,
        counterfactuals: Any,
        model: Any,
        feature_names: list[str],
        class_names: list[str] | None = None,
        raw: Any = None,
        atol: float = _ATOL,
    ):
        self.feature_names = list(feature_names)
        self.original = pd.Series(
            np.asarray(original, dtype=float), index=self.feature_names
        )
        self.counterfactuals = pd.DataFrame(
            np.atleast_2d(np.asarray(counterfactuals, dtype=float)),
            columns=self.feature_names,
        )
        self.model = model
        self.class_names = class_names
        self.raw = raw
        self._atol = atol

    def _predict(self, values: np.ndarray) -> int:
        return int(self.model.predict(np.asarray(values).reshape(1, -1))[0])

    def _label(self, cls: int) -> Any:
        if self.class_names is not None and 0 <= cls < len(self.class_names):
            return self.class_names[cls]
        return cls

    @property
    def original_prediction(self) -> int:
        """Predicted class for the original sample."""
        return self._predict(self.original.to_numpy())

    @property
    def n_counterfactuals(self) -> int:
        """Number of counterfactuals held."""
        return len(self.counterfactuals)

    def changes(self, cf_index: int | None = None) -> pd.DataFrame:
        """Return the per-feature changes for one or all counterfactuals.

        Parameters
        ----------
        cf_index : int, optional
            If given, return changes for that single counterfactual. Otherwise
            return the changes for all counterfactuals stacked, with a ``cf``
            column identifying each.

        Returns
        -------
        pd.DataFrame
            Columns: ``cf`` (omitted when ``cf_index`` given), ``feature``,
            ``original``, ``counterfactual``, ``delta``, ``direction``. Only rows
            where the feature actually changed are included, sorted by descending
            absolute change.

        """
        indices = [cf_index] if cf_index is not None else range(self.n_counterfactuals)
        frames = []
        for i in indices:
            cf = self.counterfactuals.iloc[i]
            delta = cf - self.original
            mask = delta.abs() > self._atol
            frame = pd.DataFrame(
                {
                    "cf": i,
                    "feature": self.original.index[mask],
                    "original": self.original[mask].to_numpy(),
                    "counterfactual": cf[mask].to_numpy(),
                    "delta": delta[mask].to_numpy(),
                }
            )
            frame["direction"] = np.where(frame["delta"] > 0, "increase", "decrease")
            frame = frame.reindex(
                frame["delta"].abs().sort_values(ascending=False).index
            )
            frames.append(frame)

        result = pd.concat(frames, ignore_index=True)
        if cf_index is not None:
            result = result.drop(columns="cf")
        return result

    @property
    def n_changes(self) -> list[int]:
        """Number of features changed by each counterfactual."""
        deltas = self.counterfactuals.to_numpy() - self.original.to_numpy()
        return (np.abs(deltas) > self._atol).sum(axis=1).tolist()

    @property
    def validity(self) -> float:
        """Fraction of counterfactuals that actually flip the prediction."""
        if self.n_counterfactuals == 0:
            return 0.0
        orig = self.original_prediction
        flips = sum(
            self._predict(row) != orig for row in self.counterfactuals.to_numpy()
        )
        return flips / self.n_counterfactuals

    def summary(self) -> str:
        """Return a short human-readable summary of the counterfactuals."""
        if self.n_counterfactuals == 0:
            return "No counterfactuals found."
        orig = self.original_prediction
        n = self.n_changes
        return (
            f"{self.n_counterfactuals} counterfactual(s) flipping "
            f"{self._label(orig)} → {self._label(1 - orig)}; "
            f"features changed: min={min(n)}, median={int(np.median(n))}, "
            f"max={max(n)}; validity={self.validity:.0%}."
        )

    def visualize_as_dataframe(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the underlying DiCE explanation's visualizer, if present."""
        if self.raw is None:
            raise AttributeError("No raw DiCE explanation attached to this result.")
        return self.raw.visualize_as_dataframe(*args, **kwargs)

    def __repr__(self) -> str:
        """Return the summary as the representation."""
        return f"CounterfactualResult({self.summary()})"

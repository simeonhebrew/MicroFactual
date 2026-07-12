"""Tests for counterfactual plausibility bounds, concordance, and heatmap."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from microfactual.explainability.plausibility import (
    counterfactual_concordance,
    plausible_range,
)
from microfactual.explainability.result import CounterfactualResult


class _DummyModel:
    def predict(self, X):
        return np.zeros(len(np.atleast_2d(X)), dtype=int)


def _result(original, cf, feature_names):
    return CounterfactualResult(
        original=original,
        counterfactuals=[cf],
        model=_DummyModel(),
        feature_names=feature_names,
    )


class TestPlausibleRange:
    def test_bounds_from_reference_quantiles(self):
        X = pd.DataFrame({"f0": [0.0, 1.0, 2.0, 3.0, 100.0, 200.0]})
        y = [0, 0, 0, 0, 1, 1]
        bounds = plausible_range(X, y, reference_class=0, q=(0.0, 1.0))
        # Reference class (0) spans 0..3; class 1's extremes are excluded.
        assert bounds["f0"] == [0.0, 3.0]

    def test_missing_reference_raises(self):
        X = pd.DataFrame({"f0": [1.0, 2.0]})
        with pytest.raises(ValueError):
            plausible_range(X, [0, 0], reference_class=9)


class TestConcordance:
    def test_directional_agreement(self):
        # Reference class (0) centered at 0; a "diseased" sample sits at 5.
        X = pd.DataFrame({"f0": [0.0, 0.0, 10.0, 10.0], "f1": [0.0, 0.0, 10.0, 10.0]})
        y = [0, 0, 1, 1]
        # CF decreases f0 (toward ref median 0 -> concordant) but increases f1
        # (away from ref median 0 -> discordant): expect 0.5.
        res = _result([5.0, 5.0], [1.0, 8.0], ["f0", "f1"])
        assert counterfactual_concordance(res, X, y, reference_class=0) == 0.5

    def test_nan_when_no_changes(self):
        X = pd.DataFrame({"f0": [0.0, 1.0]})
        res = _result([5.0], [5.0], ["f0"])  # no change
        assert np.isnan(counterfactual_concordance(res, X, [0, 0], reference_class=0))


class TestHeatmap:
    def _setup(self):
        X = pd.DataFrame(
            {
                "f0": [0.0, 0.5, 1.0, 9.0, 10.0, 11.0],
                "f1": [0.0, 1.0, 2.0, 8.0, 10.0, 12.0],
            }
        )
        y = [0, 0, 0, 1, 1, 1]
        res = _result([10.0, 10.0], [1.0, 2.0], ["f0", "f1"])
        return res, X, y

    def test_returns_axes(self):
        from microfactual.visualization import plot_counterfactual_heatmap

        res, X, y = self._setup()
        ax = plot_counterfactual_heatmap(
            res, X, y, reference_class=0, comparison_class=1
        )
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_empty_changes_raises(self):
        from microfactual.visualization import plot_counterfactual_heatmap

        X = pd.DataFrame({"f0": [0.0, 1.0]})
        res = _result([5.0], [5.0], ["f0"])  # nothing changed
        with pytest.raises(ValueError):
            plot_counterfactual_heatmap(res, X, [0, 0], reference_class=0)
        plt.close("all")

    def test_exports(self):
        import microfactual as mf

        for name in (
            "plausible_range",
            "counterfactual_concordance",
            "plot_counterfactual_heatmap",
        ):
            assert hasattr(mf, name)
            assert name in mf.__all__

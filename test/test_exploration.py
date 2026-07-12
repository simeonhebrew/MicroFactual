"""Tests for data-exploration / cutoff-diagnostic plots."""

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import pandas as pd
import pytest


@pytest.fixture
def abundance():
    """Features x samples abundance frame with a range of prevalence/abundance."""
    return pd.DataFrame(
        {
            "S1": [0.5, 0.01, 0.0, 0.0],
            "S2": [0.4, 0.02, 0.0, 0.0],
            "S3": [0.6, 0.00, 1e-8, 0.0],
            "S4": [0.3, 0.03, 0.0, 0.0],
        },
        index=["Common", "Mid", "Rare", "Absent"],
    )


@pytest.fixture
def dataset(abundance):
    """A MicrobiomeDataset wrapping the abundance fixture."""
    from microfactual.core import MicrobiomeDataset

    metadata = pd.DataFrame(
        {
            "Sample ID": ["S1", "S2", "S3", "S4"],
            "Group": ["A", "B", "A", "B"],
        }
    )
    return MicrobiomeDataset(abundance, metadata, target_column="Group")


class TestPerTaxonStats:
    """The underlying per-taxon statistics."""

    def test_mean_and_prevalence(self, abundance):
        from microfactual.visualization.exploration import _per_taxon_stats

        mean_ab, prev = _per_taxon_stats(abundance)
        # "Common" present in all 4 samples, "Absent" in none.
        assert prev[0] == pytest.approx(1.0)
        assert prev[3] == pytest.approx(0.0)
        assert prev[2] == pytest.approx(0.25)  # Rare present in 1/4
        assert mean_ab[0] > mean_ab[1] > mean_ab[2]


class TestPlotters:
    """Individual plot functions return axes and honour cutoffs."""

    def test_abundance_histogram(self, abundance):
        from microfactual.visualization.exploration import plot_abundance_histogram

        ax = plot_abundance_histogram(abundance, abundance_cutoff=1e-6)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_prevalence_histogram(self, abundance):
        from microfactual.visualization.exploration import plot_prevalence_histogram

        ax = plot_prevalence_histogram(abundance, prevalence_cutoff=0.5)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_prevalence_abundance_scatter(self, abundance):
        from microfactual.visualization.exploration import plot_prevalence_abundance

        ax = plot_prevalence_abundance(
            abundance, abundance_cutoff=1e-6, prevalence_cutoff=0.5
        )
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_cutoffs_optional(self, abundance):
        from microfactual.visualization.exploration import plot_abundance_histogram

        ax = plot_abundance_histogram(abundance, abundance_cutoff=None)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


class TestExplore:
    """The top-level explore() panel."""

    def test_explore_from_dataset(self, dataset):
        import microfactual as mf

        fig = mf.explore(dataset)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        plt.close("all")

    def test_explore_from_dataframe(self, abundance):
        import microfactual as mf

        fig = mf.explore(abundance, abundance_cutoff=1e-6, prevalence_cutoff=0.5)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_explore_rejects_bad_input(self):
        import microfactual as mf

        with pytest.raises(TypeError):
            mf.explore(["not", "a", "frame"])

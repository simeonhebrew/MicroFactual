"""Counterfactual explanations using DiCE."""

from typing import Any

import numpy as np
import pandas as pd

try:
    import dice_ml
except ImportError:
    dice_ml = None

from microfactual.explainability.base import BaseExplainer
from microfactual.explainability.result import (
    CounterfactualResult,
    sparsify_counterfactual,
)


class DiCEExplainer(BaseExplainer):
    """Wrapper for DiCE (Diverse Counterfactual Explanations).

    Generates counterfactual examples: "What changes to the input would
    flip the model prediction?"
    """

    def __init__(
        self,
        model: Any,
        background_data: pd.DataFrame,
        target_column: str,
        target_data: pd.Series | pd.DataFrame | None = None,
        backend: str = 'sklearn',
        continuous_features: list[str] | None = None,
        **kwargs,
    ):
        """Initialize DiCE explainer.

        Parameters
        ----------
        model : Any
            The trained model (sklearn compatible).
        background_data : pd.DataFrame
            Training data (features only).
        target_column : str
            Name of the target/outcome variable.
        target_data : pd.Series or pd.DataFrame, optional
            Target values corresponding to background_data.
            If provided, it will be joined with background_data to form
            the dataset DiCE expects.
        backend : str, default='sklearn'
            Modeling backend type.
        continuous_features : list of str, optional
            List of continuous feature names. If None, inferred from data columns.
        **kwargs
            Additional arguments passed to dice_ml.Dice

        """
        if dice_ml is None:
            raise ImportError(
                "dice-ml is required for counterfactuals. "
                "Install the explainability extra with: "
                "pip install 'microfactual[explainability]'"
            )

        super().__init__(model, background_data, **kwargs)
        self.target_column = target_column

        # DiCE expects a dataframe with both features and target
        self.dice_data_df = background_data.copy()
        if target_data is not None:
            # Ensure index alignment if possible, otherwise reset index
            self.dice_data_df[target_column] = (
                target_data.values if hasattr(target_data, 'values') else target_data
            )

        if continuous_features is None:
            continuous_features = background_data.columns.tolist()

        # Initialize DiCE Data object
        self.d = dice_ml.Data(
            dataframe=self.dice_data_df,
            continuous_features=continuous_features,
            outcome_name=target_column,
        )

        # Initialize DiCE Model object
        self.m = dice_ml.Model(model=model, backend=backend)

        # Initialize DiCE explainer
        # Default method usually 'genetic' or 'random' depending on needs
        method = kwargs.pop('method', 'genetic')
        self.exp = dice_ml.Dice(self.d, self.m, method=method, **kwargs)

    def explain(
        self,
        query_instance: pd.DataFrame,
        total_CFs: int = 5,
        desired_class: str = "opposite",
        **kwargs,
    ) -> Any:
        """Generate counterfactual explanations.

        Parameters
        ----------
        query_instance : pd.DataFrame
            The instance to explain.
        total_CFs : int, default=5
            Number of counterfactuals to generate.
        desired_class : str, default="opposite"
            Desired target class ("opposite" or specific class index).
        **kwargs
            Additional parameters passed to exp.generate_counterfactuals()

        Returns
        -------
        dice_ml.counterfactual_explanations.CounterfactualExplanations
            The explanation object containing counterfactuals.

        """
        # Ensure query instance doesn't have the target column if it's there
        if self.target_column in query_instance.columns:
            query_instance = query_instance.drop(columns=[self.target_column])

        dice_exp = self.exp.generate_counterfactuals(
            query_instance, total_CFs=total_CFs, desired_class=desired_class, **kwargs
        )
        return dice_exp


def explain_counterfactual(
    model: Any,
    query: pd.DataFrame,
    background_data: pd.DataFrame,
    y: pd.Series | pd.DataFrame | Any,
    *,
    target_column: str = "outcome",
    total_CFs: int = 5,
    desired_class: str = "opposite",
    features_to_vary: str | list[str] = "all",
    permitted_range: dict[str, list[float]] | None = None,
    sparse: bool = True,
    class_names: list[str] | None = None,
    continuous_features: list[str] | None = None,
    method: str = "genetic",
    backend: str = "sklearn",
    return_raw: bool = False,
    **kwargs: Any,
) -> Any:
    """Generate actionable counterfactual explanations for one or more samples.

    This is MicroFactual's headline entry point: it answers *"what minimal
    change in feature values would flip this sample's prediction?"* in a single
    call, wrapping :class:`DiCEExplainer`.

    Two features make the result actionable rather than a wall of numbers:

    - ``features_to_vary`` / ``permitted_range`` restrict the search to the taxa
      you consider modifiable and to plausible value ranges;
    - ``sparse=True`` (default) post-processes each counterfactual to a
      near-minimal set of changes that still flips the prediction, so you get
      *"change these few taxa"* instead of *"change everything"*.

    Because DiCE searches for counterfactuals in the model's *input* space, pass
    ``model``, ``query`` and ``background_data`` in the **same feature space the
    model was trained on** (e.g. CLR-transformed features, if the model was fit
    on CLR-transformed data).

    Parameters
    ----------
    model : Any
        A fitted, sklearn-compatible classifier (e.g. a
        :class:`~microfactual.models.classifiers.MicrobiomeClassifier` or a bare
        sklearn estimator) exposing ``predict``/``predict_proba``.
    query : pd.DataFrame
        Sample(s) to explain, as one or more rows with the same feature columns
        as ``background_data``.
    background_data : pd.DataFrame
        Training features (samples x features) used to describe the data
        distribution DiCE samples from.
    y : pd.Series, pd.DataFrame, or array-like
        Training targets aligned with ``background_data``.
    target_column : str, default="outcome"
        Name DiCE uses for the outcome column internally. Only needs to differ
        from the feature names; it is not a column in ``query``.
    total_CFs : int, default=5
        Number of counterfactuals to generate per query sample.
    desired_class : str, default="opposite"
        Target class for the counterfactuals ("opposite" flips the prediction).
    features_to_vary : str or list of str, default="all"
        Which features DiCE may change. Pass a list to restrict the search to
        modifiable taxa (e.g. diet-associated genera).
    permitted_range : dict, optional
        Per-feature ``{name: [min, max]}`` bounds constraining counterfactual
        values to a plausible range.
    sparse : bool, default=True
        If True, greedily reduce each counterfactual to a near-minimal set of
        changes that preserves the flipped prediction.
    class_names : list of str, optional
        Human-readable class labels (indexed by integer class) for the summary.
    continuous_features : list of str, optional
        Feature names to treat as continuous. Defaults to all columns of
        ``background_data`` (appropriate for abundance / CLR features).
    method : str, default="genetic"
        DiCE search method ("genetic", "random", or "kdtree").
    backend : str, default="sklearn"
        DiCE model backend.
    return_raw : bool, default=False
        If True, return the raw ``dice_ml`` explanation object instead of a
        :class:`~microfactual.explainability.result.CounterfactualResult`.
    **kwargs
        Additional arguments forwarded to
        ``dice_ml.Dice.generate_counterfactuals``.

    Returns
    -------
    CounterfactualResult or list of CounterfactualResult
        One result per query sample (a single result if ``query`` has one row).
        Each exposes ``.changes()``, ``.n_changes``, ``.validity`` and
        ``.summary()``. If ``return_raw=True``, the raw DiCE object is returned.

    Raises
    ------
    ImportError
        If the optional ``explainability`` extra (dice-ml) is not installed.

    Examples
    --------
    >>> import microfactual as mf
    >>> model = mf.MicrobiomeClassifier(preprocessing=None).fit(X_clr, y)
    >>> cf = mf.explain_counterfactual(model, X_clr.iloc[[0]], X_clr, y)
    >>> print(cf.summary())  # doctest: +SKIP
    >>> cf.changes()  # doctest: +SKIP

    """
    explainer = DiCEExplainer(
        model=model,
        background_data=background_data,
        target_column=target_column,
        target_data=y,
        continuous_features=continuous_features,
        backend=backend,
        method=method,
    )
    raw = explainer.explain(
        query,
        total_CFs=total_CFs,
        desired_class=desired_class,
        features_to_vary=features_to_vary,
        permitted_range=permitted_range,
        **kwargs,
    )
    if return_raw:
        return raw

    feature_names = list(background_data.columns)
    # Iterate over query rows (not over cf_examples_list): DiCE may return fewer
    # examples than queries — or an empty list — when it finds no counterfactual.
    # We must still return one CounterfactualResult per query row.
    cf_examples = list(getattr(raw, "cf_examples_list", None) or [])
    n_queries = len(query)
    results = []
    for idx in range(n_queries):
        original = query.iloc[idx][feature_names].to_numpy(dtype=float)
        cfs_df = cf_examples[idx].final_cfs_df if idx < len(cf_examples) else None
        if cfs_df is None or len(cfs_df) == 0:
            cfs = np.empty((0, len(feature_names)))
        else:
            cfs = cfs_df[feature_names].to_numpy(dtype=float)
            if sparse:
                cfs = np.vstack(
                    [sparsify_counterfactual(original, row, model) for row in cfs]
                )
        results.append(
            CounterfactualResult(
                original=original,
                counterfactuals=cfs,
                model=model,
                feature_names=feature_names,
                class_names=class_names,
                raw=raw,
            )
        )

    # Always return a single result for a single-row query (never a bare list).
    return results[0] if n_queries == 1 else results


def counterfactual_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | Any,
    *,
    background_data: pd.DataFrame | None = None,
    target_column: str = "outcome",
    total_CFs: int = 3,
    desired_class: str = "opposite",
    features_to_vary: str | list[str] = "all",
    permitted_range: dict[str, list[float]] | None = None,
    sparse: bool = True,
    method: str = "genetic",
    backend: str = "sklearn",
    top_n: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Aggregate counterfactuals across a cohort into per-taxon importance.

    Generates counterfactuals for every sample in ``X`` and summarises how often
    each taxon has to change to flip predictions, and in which direction. This
    is a *local-aggregated* importance: unlike a global feature-importance score,
    it reflects the taxa that actually drive individual decisions across the
    cohort, with an interpretable direction of change.

    As with :func:`explain_counterfactual`, pass everything in the model's input
    feature space (e.g. CLR).

    Parameters
    ----------
    model : Any
        Fitted, sklearn-compatible classifier.
    X : pd.DataFrame
        Samples to explain (samples x features).
    y : pd.Series, pd.DataFrame, or array-like
        Targets aligned with ``background_data``.
    background_data : pd.DataFrame, optional
        Distribution DiCE samples from; defaults to ``X`` itself.
    target_column : str, default="outcome"
        See :func:`explain_counterfactual`.
    total_CFs : int, default=3
        Counterfactuals generated per sample before aggregation.
    desired_class : str, default="opposite"
        See :func:`explain_counterfactual`.
    features_to_vary : str or list of str, default="all"
        See :func:`explain_counterfactual`.
    permitted_range : dict, optional
        See :func:`explain_counterfactual`.
    sparse : bool, default=True
        See :func:`explain_counterfactual`.
    method : str, default="genetic"
        See :func:`explain_counterfactual`.
    backend : str, default="sklearn"
        See :func:`explain_counterfactual`.
    top_n : int, optional
        If given, return only the ``top_n`` most frequently implicated taxa.
    **kwargs
        Forwarded to ``dice_ml.Dice.generate_counterfactuals``.

    Returns
    -------
    pd.DataFrame
        One row per implicated taxon, sorted by how many samples implicate it.
        Columns: ``feature``, ``n_samples`` (samples whose counterfactuals change
        it), ``frequency`` (fraction of explained samples), ``mean_delta`` (mean
        signed change, averaged per sample), ``direction`` (net "increase"/
        "decrease").

    """
    background = X if background_data is None else background_data

    # One (sample, feature) record per implicated taxon, holding its per-sample
    # mean signed change across that sample's counterfactuals.
    records: list[tuple[str, float]] = []
    n_explained = 0
    for i in range(len(X)):
        try:
            result = explain_counterfactual(
                model,
                X.iloc[[i]],
                background,
                y,
                target_column=target_column,
                total_CFs=total_CFs,
                desired_class=desired_class,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range,
                sparse=sparse,
                method=method,
                backend=backend,
                **kwargs,
            )
        except Exception:
            # A single sample failing to yield counterfactuals shouldn't abort
            # the whole cohort sweep.
            continue

        changes = result.changes()
        if len(changes) == 0:
            continue
        n_explained += 1
        per_feature = changes.groupby("feature")["delta"].mean()
        records.extend(per_feature.items())

    columns = ["feature", "n_samples", "frequency", "mean_delta", "direction"]
    if not records:
        return pd.DataFrame(columns=columns)

    per_sample = pd.DataFrame(records, columns=["feature", "delta"])
    agg = per_sample.groupby("feature")["delta"].agg(["size", "mean"])
    agg.columns = ["n_samples", "mean_delta"]
    agg["frequency"] = agg["n_samples"] / n_explained
    agg["direction"] = np.where(agg["mean_delta"] > 0, "increase", "decrease")
    agg = agg.sort_values(
        ["n_samples", "mean_delta"], ascending=[False, False]
    ).reset_index()
    agg = agg[columns]
    if top_n is not None:
        agg = agg.head(top_n)
    return agg

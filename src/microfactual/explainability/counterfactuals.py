"""Counterfactual explanations using DiCE."""

from typing import Any

import pandas as pd

try:
    import dice_ml
except ImportError:
    dice_ml = None

from microfactual.explainability.base import BaseExplainer


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
    continuous_features: list[str] | None = None,
    method: str = "genetic",
    backend: str = "sklearn",
    **kwargs: Any,
) -> Any:
    """Generate counterfactual explanations for one or more samples.

    This is MicroFactual's headline entry point: it answers *"what minimal
    change in feature values would flip this sample's prediction?"* in a single
    call, wrapping :class:`DiCEExplainer`.

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
    continuous_features : list of str, optional
        Feature names to treat as continuous. Defaults to all columns of
        ``background_data`` (appropriate for abundance / CLR features).
    method : str, default="genetic"
        DiCE search method ("genetic", "random", or "kdtree").
    backend : str, default="sklearn"
        DiCE model backend.
    **kwargs
        Additional arguments forwarded to
        ``dice_ml.Dice.generate_counterfactuals``.

    Returns
    -------
    dice_ml.counterfactual_explanations.CounterfactualExplanations
        The explanation object. Inspect it with
        ``result.visualize_as_dataframe(show_only_changes=True)`` or read the
        raw counterfactuals from ``result.cf_examples_list``.

    Raises
    ------
    ImportError
        If the optional ``explainability`` extra (dice-ml) is not installed.

    Examples
    --------
    >>> import microfactual as mf
    >>> model = mf.MicrobiomeClassifier(preprocessing=None).fit(X_clr, y)
    >>> cf = mf.explain_counterfactual(model, X_clr.iloc[[0]], X_clr, y)
    >>> cf.visualize_as_dataframe(show_only_changes=True)  # doctest: +SKIP

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
    return explainer.explain(
        query, total_CFs=total_CFs, desired_class=desired_class, **kwargs
    )

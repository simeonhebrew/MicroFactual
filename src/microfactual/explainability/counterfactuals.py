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

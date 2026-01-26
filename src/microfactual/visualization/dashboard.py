"""Wrapper for ExplainerDashboard integration."""

import pandas as pd
import warnings
from typing import Any, Optional

try:
    from explainerdashboard import ClassifierExplainer, ExplainerDashboard
except ImportError:
    ClassifierExplainer = None
    ExplainerDashboard = None


def launch_dashboard(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | Any,
    run: bool = True,
    **kwargs
) -> Any:
    """Launch ExplainerDashboard for a trained model.
    
    Parameters
    ----------
    model : Any
        Trained model or MicrobiomeClassifier instance.
    X : pd.DataFrame
        Detailed input data (features).
    y : pd.Series or array-like
        Detailed target labels.
    run : bool, default=True
        Whether to start the dashboard server immediately.
    **kwargs
        Additional arguments passed to ExplainerDashboard.run().
        
    Returns
    -------
    ExplainerDashboard
        The dashboard instance.
    """
    if ClassifierExplainer is None:
        raise ImportError(
            "explainerdashboard is required. "
            "Install it with: pip install explainerdashboard"
        )

    # Sanitize column names for SHAP compatibility
    # SHAP often fails with special characters in feature names
    if hasattr(X, "columns"):
        X = X.copy()
        X.columns = X.columns.str.replace(r"[.{}]", "_", regex=True)

    # Create explainer
    # If using MicrobiomeClassifier, we can pass it directly 
    # as it behaves like an sklearn estimator
    explainer = ClassifierExplainer(model, X, y)
    
    # Create dashboard
    db = ExplainerDashboard(explainer, header_hide_selector=True)
    
    if run:
        db.run(**kwargs)
        
    return db

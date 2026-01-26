"""Base interface for all explanation methods.

Decouples the core library from specific explainability frameworks.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Optional

class BaseExplainer(ABC):
    """Abstract base class for model explainers."""
    
    def __init__(self, model: Any, data: pd.DataFrame, **kwargs):
        """Initialize the explainer.
        
        Parameters
        ----------
        model : Any
            The trained model to explain.
        data : pd.DataFrame
            Background/training data used for the explanation method.
        **kwargs
            Additional framework-specific parameters.
        """
        self.model = model
        self.data = data
        self.kwargs = kwargs
        
    @abstractmethod
    def explain(self, query_instance: pd.DataFrame, **kwargs) -> Any:
        """Generate explanations for a query instance.
        
        Parameters
        ----------
        query_instance : pd.DataFrame
            Instance(s) to explain.
        **kwargs
            Additional parameters for generating the explanation.
            
        Returns
        -------
        Any
            The explanation object (framework specific).
        """
        pass

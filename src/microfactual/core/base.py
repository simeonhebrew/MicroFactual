"""Base classes and interfaces for microfactual models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class BaseModel(ABC):
    """Base interface for all machine learning models in microfactual.
    
    This abstract base class defines the common interface that all models
    (regardless of backend - sklearn, pytorch, etc.) must implement.
    This ensures consistency and makes it easy to swap between different
    model implementations.
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Train the model on the provided data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with samples as rows, features as columns
        y : pd.Series
            Target vector with encoded labels
            
        Returns
        -------
        BaseModel
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions on new data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with samples as rows, features as columns
            
        Returns
        -------
        pd.Series
            Predicted class labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities for each class.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with samples as rows, features as columns
            
        Returns
        -------
        pd.DataFrame
            Prediction probabilities with samples as rows, classes as columns
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with samples as rows, features as columns
        y : pd.Series
            True target labels
            
        Returns
        -------
        dict
            Dictionary containing evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the trained model to disk.
        
        Parameters
        ----------
        filepath : str
            Path where the model should be saved
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load a trained model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model file
            
        Returns
        -------
        BaseModel
            Loaded model instance
        """
        pass
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance scores if available.
        
        This is an optional method that models can implement if they
        provide feature importance information.
        
        Returns
        -------
        pd.Series or None
            Feature importance scores, or None if not available
        """
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        This is an optional method for introspection.
        
        Returns
        -------
        dict
            Dictionary of model parameters
        """
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters.
        
        This is an optional method for parameter tuning.
        
        Parameters
        ----------
        **params
            Model parameters to set
            
        Returns
        -------
        BaseModel
            Self for method chaining
        """
        return self
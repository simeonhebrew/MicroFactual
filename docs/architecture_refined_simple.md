# Refined Simplified Architecture

## Core Philosophy
Start with sklearn backend + configuration files. Add complexity only when needed.

## Directory Structure (Minimal Changes)

```
microfactual/
├── __init__.py                 # Main API exports
├── core/
│   ├── __init__.py
│   ├── dataset.py             # MicrobiomeDataset class
│   └── base.py                # BaseModel interface
├── models/
│   ├── __init__.py
│   ├── sklearn_backend.py     # SklearnModel + ModelFactory
│   └── model_registry.py      # Available models registry
├── preprocessing/
│   ├── __init__.py
│   └── pipeline.py            # Configurable preprocessing
├── data_processing.py          # Keep your current functions
├── modeling.py                 # Keep + enhance with new backend
├── visualization.py            # Keep current functions  
├── utils.py                    # Keep current functions
├── main.py                     # Enhanced CLI + API
└── configs/                    # Configuration templates
    ├── classification.yaml
    ├── regression.yaml
    └── discovery.yaml
```

## Core Components Implementation

### 1. MicrobiomeDataset Class

```python
# microfactual/core/dataset.py

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

class MicrobiomeDataset:
    """Central data structure for microbiome datasets."""
    
    def __init__(self, abundance: pd.DataFrame, metadata: pd.DataFrame, 
                 target_column: str, sample_column: str = "Sample ID"):
        self.abundance = abundance  # features x samples
        self.metadata = metadata
        self.target_column = target_column
        self.sample_column = sample_column
        self._validate_data()
    
    @classmethod
    def from_files(cls, abundance_file: str, metadata_file: str,
                   target_column: str, sample_column: str = "Sample ID"):
        """Load dataset from files (use your existing load_data logic)."""
        # Use your existing load_data function here
        abundance = pd.read_csv(abundance_file, sep='\t', index_col='Species')
        metadata = pd.read_csv(metadata_file, sep='\t')
        return cls(abundance, metadata, target_column, sample_column)
    
    @property
    def X(self) -> pd.DataFrame:
        """Features matrix (samples x features) - ready for sklearn."""
        return self.abundance.T  # Transpose to samples x features
    
    @property  
    def y(self) -> pd.Series:
        """Target vector."""
        return self.metadata[self.target_column].astype('category').cat.codes
    
    def _validate_data(self) -> None:
        """Basic data validation."""
        # Add your validation logic here
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        return {
            'n_samples': self.abundance.shape[1],
            'n_features': self.abundance.shape[0],
            'target_classes': self.metadata[self.target_column].unique().tolist(),
            'sparsity': (self.abundance == 0).mean().mean()
        }
```

### 2. BaseModel Interface

```python
# microfactual/core/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

class BaseModel(ABC):
    """Base interface for all models regardless of backend."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities."""
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load model from disk."""
        pass
```

### 3. Sklearn Backend Implementation

```python
# microfactual/models/sklearn_backend.py

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, Any, Optional

from ..core.base import BaseModel

class SklearnModel(BaseModel):
    """Sklearn model wrapper that implements BaseModel interface."""
    
    def __init__(self, model_type: str, parameters: Optional[Dict[str, Any]] = None,
                 cv_params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.parameters = parameters or {}
        self.cv_params = cv_params or {'cv': 5}
        self.model = None
        self.is_fitted = False
        
        # Initialize base model
        self._init_model()
    
    def _init_model(self):
        """Initialize the sklearn model based on type."""
        model_registry = {
            'RandomForestClassifier': RandomForestClassifier,
            'SVC': SVC,
            # Add more as needed
        }
        
        if self.model_type not in model_registry:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        base_model = model_registry[self.model_type](**self.parameters)
        
        # Wrap in GridSearchCV if hyperparameters provided
        if 'param_grid' in self.cv_params:
            self.model = GridSearchCV(base_model, 
                                    self.cv_params['param_grid'],
                                    cv=self.cv_params.get('cv', 5))
        else:
            self.model = base_model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SklearnModel':
        """Train the model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return pd.Series(self.model.predict(X), index=X.index)
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        proba = self.model.predict_proba(X)
        return pd.DataFrame(proba, index=X.index, 
                          columns=[f'class_{i}' for i in range(proba.shape[1])])
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X).iloc[:, 1]  # Positive class probability
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y, y_proba) if len(y.unique()) == 2 else None
        }
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'SklearnModel':
        """Load model from disk."""
        model_instance = cls.__new__(cls)  # Create without __init__
        model_instance.model = joblib.load(filepath)
        model_instance.is_fitted = True
        return model_instance

# Model Factory
def create_model(config: Dict[str, Any]) -> BaseModel:
    """Create model from configuration."""
    backend = config.get('backend', 'sklearn')
    
    if backend == 'sklearn':
        return SklearnModel(
            model_type=config['type'],
            parameters=config.get('parameters', {}),
            cv_params=config.get('cv_params', {})
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
```

### 4. Configuration Files

```yaml
# microfactual/configs/classification.yaml

preprocessing:
  steps:
    - name: filter_data
      parameters:
        abundance_cutoff: 1e-6
        prevalence_cutoff: 0.05
    - name: clr_transform
      parameters:
        log_n0: 1e-6

model:
  backend: sklearn
  type: RandomForestClassifier
  parameters:
    n_estimators: 100
    random_state: 42
  cv_params:
    cv: 5
    param_grid:
      max_features: [1, 5, 10, 'sqrt', 'log2']

evaluation:
  metrics: [accuracy, f1_score, roc_auc]
  cross_validation: 
    cv: 5
    scoring: [accuracy, f1]

output:
  save_model: true
  save_predictions: true
  save_plots: [roc_curve, feature_importance]
```

### 5. Enhanced Main API

```python
# microfactual/main.py (enhanced)

import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from .core.dataset import MicrobiomeDataset
from .models.sklearn_backend import create_model
# Keep your existing imports and functions

def run_pipeline_from_config(config_path: str, 
                           abundance_file: str,
                           metadata_file: str,
                           target_column: str,
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run pipeline using configuration file."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    dataset = MicrobiomeDataset.from_files(
        abundance_file=abundance_file,
        metadata_file=metadata_file,
        target_column=target_column
    )
    
    # Apply preprocessing (use your existing functions)
    X, y = dataset.X, dataset.y
    # Apply preprocessing steps from config...
    
    # Create and train model
    model = create_model(config['model'])
    model.fit(X, y)
    
    # Evaluate
    results = model.evaluate(X, y)
    
    # Save outputs based on config
    if output_dir:
        if config['output'].get('save_model', False):
            model.save(f"{output_dir}/model.pkl")
        # Add other saving logic...
    
    return {
        'model': model,
        'results': results,
        'dataset_info': dataset.get_info()
    }

# Keep your existing run_pipeline function for backward compatibility
def classify(abundance_file: str, metadata_file: str, target_column: str,
             model_type: str = 'RandomForestClassifier', 
             config: Optional[str] = None,
             output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Simple one-function classification API."""
    
    if config:
        return run_pipeline_from_config(config, abundance_file, 
                                      metadata_file, target_column, output_dir)
    else:
        # Use default configuration
        default_config = {
            'model': {
                'backend': 'sklearn',
                'type': model_type,
                'parameters': {'n_estimators': 100, 'random_state': 42}
            }
        }
        # Create temporary config and run...
```

### 6. Enhanced CLI

```python
# Enhanced CLI in main.py

def main():
    parser = argparse.ArgumentParser(description="Microbiome ML Pipeline")
    
    # Keep existing arguments for backward compatibility
    parser.add_argument("--abundance", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    
    # Add new configuration option
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model-type", type=str, default="RandomForestClassifier",
                       choices=["RandomForestClassifier", "SVC"])
    
    args = parser.parse_args()
    
    if args.config:
        results = run_pipeline_from_config(
            config_path=args.config,
            abundance_file=args.abundance,
            metadata_file=args.metadata, 
            target_column=args.target,
            output_dir=args.output_dir
        )
    else:
        # Use existing function for backward compatibility
        run_pipeline(
            abundance_path=args.abundance,
            metadata_path=args.metadata,
            target_column=args.target,
            output_dir=args.output_dir
        )
```

## Usage Examples

### 1. Backward Compatible (Current Users)
```bash
# Existing usage still works
microfactual --abundance data.txt --metadata meta.txt --target disease
```

### 2. New Configuration-Based
```bash
# With config file
microfactual --abundance data.txt --metadata meta.txt --target disease --config configs/classification.yaml

# Or specify model type directly
microfactual --abundance data.txt --metadata meta.txt --target disease --model-type SVC
```

### 3. Python API
```python
import microfactual as mf

# Simple API (like your current system)
results = mf.classify(
    abundance_file="data.txt",
    metadata_file="meta.txt", 
    target_column="disease"
)

# With configuration
results = mf.classify(
    abundance_file="data.txt",
    metadata_file="meta.txt",
    target_column="disease",
    config="configs/classification.yaml"
)

# Direct object usage
dataset = mf.MicrobiomeDataset.from_files("data.txt", "meta.txt", "disease")
model = mf.create_model({
    'backend': 'sklearn',
    'type': 'RandomForestClassifier', 
    'parameters': {'n_estimators': 200}
})
model.fit(dataset.X, dataset.y)
```

## Implementation Phases

### Phase 1 (Week 1-2): Core Structure
1. Create `MicrobiomeDataset` class
2. Implement `BaseModel` interface and `SklearnModel`
3. Create basic config file support
4. Ensure backward compatibility

### Phase 2 (Week 3): Enhancement  
1. Add 2-3 more sklearn models
2. Enhanced preprocessing pipeline
3. Better configuration validation

### Phase 3 (Week 4): Polish
1. Documentation
2. Example config files
3. Error handling improvements
4. Testing

## Why This Works Well

1. **Immediate Value**: Current users can keep using existing interface
2. **Clear Growth Path**: Easy to add PyTorch backend later by implementing BaseModel
3. **Configuration Flexibility**: Power users can customize everything via YAML
4. **Simple Testing**: Each component can be tested independently  
5. **Professional Structure**: Clean interfaces without over-engineering

**Does this refined approach capture what you were thinking? Should we start implementing the core components?**
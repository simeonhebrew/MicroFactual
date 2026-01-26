"""Central data structure for microbiome datasets."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# Import your existing data processing functions
from ..data_processing import load_data, filter_data, clr_transform


class MicrobiomeDataset:
    """Central data structure for microbiome datasets.
    
    This class provides a clean interface for microbiome data, building on
    the existing data processing functions while providing sklearn-compatible
    data access patterns.
    
    Attributes:
        abundance (pd.DataFrame): Abundance data (features x samples)
        metadata (pd.DataFrame): Sample metadata  
        target_column (str): Name of target variable column
        sample_column (str): Name of sample ID column
    """
    
    def __init__(self, 
                 abundance: pd.DataFrame, 
                 metadata: pd.DataFrame,
                 target_column: str,
                 sample_column: str = "Sample ID"):
        """Initialize MicrobiomeDataset.
        
        Parameters
        ----------
        abundance : pd.DataFrame
            Abundance data with features as rows, samples as columns
        metadata : pd.DataFrame  
            Sample metadata with sample information
        target_column : str
            Column name for target variable in metadata
        sample_column : str, optional
            Column name for sample IDs in metadata, by default "Sample ID"
        """
        self.abundance = abundance.copy()
        self.metadata = metadata.copy()
        self.target_column = target_column
        self.sample_column = sample_column
        
        # Store original data for reference
        self._original_abundance = abundance.copy()
        self._original_metadata = metadata.copy()
        
        # Track preprocessing steps applied
        self._preprocessing_history = []
        
        # Validate data consistency
        self._validate_data()
    
    @classmethod
    def from_files(cls, 
                   abundance_file: str, 
                   metadata_file: str,
                   target_column: str,
                   sample_column: str = "Sample ID") -> 'MicrobiomeDataset':
        """Create MicrobiomeDataset from files using existing load_data function.
        
        Parameters
        ----------
        abundance_file : str
            Path to abundance data file (tab-separated)
        metadata_file : str
            Path to metadata file (tab-separated)
        target_column : str
            Column name for target variable
        sample_column : str, optional
            Column name for sample IDs, by default "Sample ID"
            
        Returns
        -------
        MicrobiomeDataset
            Initialized dataset object
        """
        # Use existing load_data function but get the aligned data
        abundance, labels = load_data(
            abundance_file=abundance_file,
            metadata_file=metadata_file,
            sample_column=sample_column,
            target_column=target_column
        )
        
        # Load metadata separately to preserve all columns
        metadata = pd.read_csv(metadata_file, sep='\t')
        
        # Filter metadata to match the samples in abundance data
        metadata_clean = metadata.dropna(subset=[target_column])
        metadata_aligned = metadata_clean[
            metadata_clean[sample_column].isin(abundance.columns)
        ].copy()
        
        return cls(
            abundance=abundance,
            metadata=metadata_aligned, 
            target_column=target_column,
            sample_column=sample_column
        )
    
    @property
    def X(self) -> pd.DataFrame:
        """Features matrix in sklearn format (samples x features).
        
        Returns
        -------
        pd.DataFrame
            Transposed abundance matrix with samples as rows, features as columns
        """
        return self.abundance.T  # Transpose to samples x features for sklearn
    
    @property
    def y(self) -> pd.Series:
        """Target vector with encoded categorical labels.
        
        Returns
        -------
        pd.Series
            Encoded target labels as categorical codes
        """
        # Align metadata with abundance samples
        sample_order = self.abundance.columns
        metadata_aligned = self.metadata.set_index(self.sample_column).loc[sample_order]
        
        target_series = metadata_aligned[self.target_column].astype('category')
        return target_series.cat.codes
    
    @property
    def target_names(self) -> list:
        """Get the original target class names.
        
        Returns
        -------
        list
            List of unique target class names
        """
        return self.metadata[self.target_column].astype('category').cat.categories.tolist()
    
    @property
    def sample_names(self) -> pd.Index:
        """Get sample names/IDs.
        
        Returns
        -------
        pd.Index
            Sample IDs from abundance data columns
        """
        return self.abundance.columns
    
    @property
    def feature_names(self) -> pd.Index:
        """Get feature names (e.g., species names).
        
        Returns
        -------
        pd.Index
            Feature names from abundance data index
        """
        return self.abundance.index
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information.
        
        Returns
        -------
        dict
            Dictionary containing dataset statistics and metadata
        """
        info = {
            # Basic dimensions
            'n_samples': self.abundance.shape[1],
            'n_features': self.abundance.shape[0],
            'n_metadata_columns': len(self.metadata.columns),
            
            # Target information
            'target_column': self.target_column,
            'target_classes': self.target_names,
            'class_distribution': self.metadata[self.target_column].value_counts().to_dict(),
            
            # Data quality metrics
            'sparsity': (self.abundance == 0).sum().sum() / self.abundance.size,
            'mean_reads_per_sample': self.abundance.sum(axis=0).mean(),
            'std_reads_per_sample': self.abundance.sum(axis=0).std(),
            
            # Feature statistics
            'mean_features_per_sample': (self.abundance > 0).sum(axis=0).mean(),
            'features_present_all_samples': ((self.abundance > 0).all(axis=1)).sum(),
            'features_present_no_samples': ((self.abundance == 0).all(axis=1)).sum(),
            
            # Preprocessing history
            'preprocessing_steps': self._preprocessing_history.copy()
        }
        
        return info
    
    def filter_features(self, 
                       abundance_cutoff: float = 1e-6,
                       prevalence_cutoff: float = 0.05,
                       inplace: bool = True) -> Optional['MicrobiomeDataset']:
        """Apply feature filtering using existing filter_data function.
        
        Parameters
        ----------
        abundance_cutoff : float, optional
            Minimum mean abundance threshold, by default 1e-6
        prevalence_cutoff : float, optional
            Minimum prevalence threshold, by default 0.05
        inplace : bool, optional
            Whether to modify dataset in place, by default True
            
        Returns
        -------
        MicrobiomeDataset or None
            Filtered dataset if inplace=False, None otherwise
        """
        filtered_abundance = filter_data(
            self.abundance, 
            abundance_cutoff=abundance_cutoff,
            prevalence_cutoff=prevalence_cutoff
        )
        
        # Track preprocessing step
        step_info = {
            'step': 'filter_features',
            'parameters': {
                'abundance_cutoff': abundance_cutoff,
                'prevalence_cutoff': prevalence_cutoff
            },
            'features_before': self.abundance.shape[0],
            'features_after': filtered_abundance.shape[0]
        }
        
        if inplace:
            self.abundance = filtered_abundance
            self._preprocessing_history.append(step_info)
            return None
        else:
            # Create new instance
            new_dataset = MicrobiomeDataset(
                abundance=filtered_abundance,
                metadata=self.metadata,
                target_column=self.target_column,
                sample_column=self.sample_column
            )
            new_dataset._preprocessing_history = self._preprocessing_history + [step_info]
            return new_dataset
    
    def clr_transform(self, 
                     log_n0: float = 1e-6,
                     inplace: bool = True) -> Optional['MicrobiomeDataset']:
        """Apply CLR transformation using existing clr_transform function.
        
        Parameters
        ----------
        log_n0 : float, optional
            Pseudocount for zero replacement, by default 1e-6
        inplace : bool, optional
            Whether to modify dataset in place, by default True
            
        Returns
        -------
        MicrobiomeDataset or None
            Transformed dataset if inplace=False, None otherwise
        """
        clr_data = clr_transform(self.abundance, log_n0=log_n0)
        
        # Track preprocessing step
        step_info = {
            'step': 'clr_transform',
            'parameters': {'log_n0': log_n0}
        }
        
        if inplace:
            self.abundance = clr_data.T  # clr_transform returns samples x features, we want features x samples
            self._preprocessing_history.append(step_info)
            return None
        else:
            # Create new instance  
            new_dataset = MicrobiomeDataset(
                abundance=clr_data.T,  # Transpose back to features x samples
                metadata=self.metadata,
                target_column=self.target_column,
                sample_column=self.sample_column
            )
            new_dataset._preprocessing_history = self._preprocessing_history + [step_info]
            return new_dataset
    
    def reset_preprocessing(self) -> None:
        """Reset to original data, removing all preprocessing steps."""
        self.abundance = self._original_abundance.copy()
        self.metadata = self._original_metadata.copy()
        self._preprocessing_history = []
    
    def _validate_data(self) -> None:
        """Validate data consistency and quality.
        
        Raises
        ------
        ValueError
            If data validation fails
        """
        # Check if target column exists
        if self.target_column not in self.metadata.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in metadata")
        
        # Check if sample column exists
        if self.sample_column not in self.metadata.columns:
            raise ValueError(f"Sample column '{self.sample_column}' not found in metadata")
        
        # Check for missing target values
        if self.metadata[self.target_column].isna().any():
            n_missing = self.metadata[self.target_column].isna().sum()
            print(f"Warning: {n_missing} samples have missing target values")
        
        # Check sample alignment
        abundance_samples = set(self.abundance.columns)
        metadata_samples = set(self.metadata[self.sample_column])
        
        if not abundance_samples.issubset(metadata_samples):
            missing_in_metadata = abundance_samples - metadata_samples
            print(f"Warning: {len(missing_in_metadata)} samples in abundance data not found in metadata")
        
        # Check for negative values in abundance (should be non-negative)
        if (self.abundance < 0).any().any():
            print("Warning: Negative values found in abundance data")
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (f"MicrobiomeDataset(n_samples={self.abundance.shape[1]}, "
                f"n_features={self.abundance.shape[0]}, "
                f"target='{self.target_column}', "
                f"preprocessing_steps={len(self._preprocessing_history)})")
    
    def __str__(self) -> str:
        """Detailed string description of the dataset."""
        info = self.get_info()
        return f"""MicrobiomeDataset Summary:
  Samples: {info['n_samples']}
  Features: {info['n_features']}
  Target: {info['target_column']} ({len(info['target_classes'])} classes)
  Sparsity: {info['sparsity']:.2%}
  Preprocessing steps: {len(info['preprocessing_steps'])}"""
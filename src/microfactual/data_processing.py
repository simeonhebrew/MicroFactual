"""Data processing utilities for microfactual."""

import numpy as np
import pandas as pd


def load_data(
    abundance_file: str,
    metadata_file: str,
    sample_column: str = "Sample ID",
    target_column: str = "Group",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess microbiome abundance and metadata files.

    Parameters
    ----------
    abundance_file : str
        Path to the abundance data file (tab-separated, rows=species, columns=samples).
    metadata_file : str
        Path to the metadata file (tab-separated).
    sample_column : str, optional
        Column name in metadata for sample IDs (default: "Sample ID").
    target_column : str, optional
        Column name in metadata for target variable (default: "Group").

    Returns
    -------
    abundance : pd.DataFrame
        Abundance data aligned to cleaned metadata (rows=species, columns=samples).
    sample_labels : pd.Series
        Encoded target labels as categorical codes.

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    KeyError
        If required columns are missing.

    """
    abundance = pd.read_csv(abundance_file, sep="\t", index_col="Species")
    metadata = pd.read_csv(metadata_file, sep="\t")

    # Clean metadata and get labels
    metadata_clean = metadata.dropna(subset=[target_column])
    sample_labels = metadata_clean[target_column].astype("category").cat.codes

    # Align abundance data with cleaned metadata
    abundance = abundance[metadata_clean[sample_column]]
    return abundance, sample_labels


def filter_data(
    abundance: pd.DataFrame,
    abundance_cutoff: float = 1e-6,
    prevalence_cutoff: float = 0.05,
) -> pd.DataFrame:
    """Filter features based on abundance and prevalence thresholds.

    Parameters
    ----------
    abundance : pd.DataFrame
        Abundance data (rows=species, columns=samples).
    abundance_cutoff : float, optional
        Minimum mean abundance required to retain a species (default: 1e-6).
    prevalence_cutoff : float, optional
        Minimum fraction of samples in which a species must be present (default: 0.05).
        Must be between 0 and 1.

    Returns
    -------
    pd.DataFrame
        Filtered abundance data.

    Raises
    ------
    ValueError
        If abundance_cutoff is negative or prevalence_cutoff is not in [0, 1].

    """
    if abundance_cutoff < 0:
        raise ValueError("abundance_cutoff must be non-negative")
    if not (0 <= prevalence_cutoff <= 1):
        raise ValueError("prevalence_cutoff must be between 0 and 1")
    # Abundance filter
    mean_abundance = abundance.mean(axis=1)
    abundance_filtered = abundance.loc[mean_abundance >= abundance_cutoff]

    # Prevalence filter
    prevalence = (abundance_filtered > 0).mean(axis=1)
    return abundance_filtered.loc[prevalence >= prevalence_cutoff]


def clr_transform(
    data: pd.DataFrame,
    log_n0: float = 1e-6,
) -> pd.DataFrame:
    """Apply centered log-ratio (CLR) transformation to abundance data.

    Parameters
    ----------
    data : pd.DataFrame
        Abundance data (rows=species, columns=samples).
    log_n0 : float, optional
        Value to replace zeros before log transformation (default: 1e-6).

    Returns
    -------
    pd.DataFrame
        CLR-transformed data (samples as rows, species as columns).

    """
    data = data.replace(0, log_n0)
    gm = np.exp(np.log(data).mean(axis=1))
    return np.log(data.div(gm, axis=0)).T

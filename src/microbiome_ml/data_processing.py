import pandas as pd
import numpy as np


def load_data(
    abundance_file, metadata_file, sample_column="Sample ID", target_column="Group"
):
    """Load and preprocess data"""
    abundance = pd.read_csv(abundance_file, sep="\t", index_col="Species")
    metadata = pd.read_csv(metadata_file, sep="\t")

    # Clean metadata and get labels
    metadata_clean = metadata.dropna(subset=[target_column])
    sample_labels = metadata_clean[target_column].astype("category").cat.codes

    # Align abundance data with cleaned metadata
    abundance = abundance[metadata_clean[sample_column]]
    return abundance, sample_labels


def filter_data(abundance, abundance_cutoff=1e-6, prevalence_cutoff=0.05):
    """Filter features based on abundance and prevalence"""
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


def clr_transform(data, log_n0=1e-6):
    """Centered log-ratio transformation"""
    data = data.replace(0, log_n0)
    gm = np.exp(np.log(data).mean(axis=1))
    return np.log(data.div(gm, axis=0)).T

import logging
import argparse
import os
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#%%
def load_data(abundance_file, metadata_file, target_column='Group'):
    """Load and preprocess data"""
    abundance = pd.read_csv(abundance_file, sep='\t', index_col='Species')
    metadata = pd.read_csv(metadata_file, sep='\t')

    # Clean metadata and get labels
    metadata_clean = metadata.dropna(subset=[target_column])
    sample_labels = metadata_clean[target_column].astype('category').cat.codes

    # Align abundance data with cleaned metadata
    abundance = abundance[metadata_clean['Sample ID']]
    return abundance, sample_labels

def filter_data(abundance, abundance_cutoff=1e-6, prevalence_cutoff=0.05):
    """Filter features based on abundance and prevalence"""
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

def train_model(X, y, cv_splits=2):
    """Train Random Forest classifier with cross-validation"""
    param_grid = {'max_features': np.unique(np.linspace(1, X.shape[1], 5, dtype=int))}
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = RepeatedKFold(n_splits=cv_splits, n_repeats=2, random_state=42)

    grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search

def plot_roc(y_true, y_probs, save_path=None, show=False):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    if save_path:
        save_path = os.path.join(save_path, 'roc_curve.png')
        plt.savefig(save_path)
        logging.info(f"ROC curve saved to {save_path}")
    if show:
        plt.show()


#%%
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the pipeline...")
    
    # Get the command line arguments
    parser = argparse.ArgumentParser(description="Microbiome ML Pipeline")
    parser.add_argument('--abundance', type=str, required=True, help='Path to abundance data file')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--target', type=str, default='Group', help='Target variable in metadata')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory for saving results')
    args = parser.parse_args()
    abundance_path = args.abundance
    metadata_path = args.metadata
    target_column = args.target
    output_dir = args.output_dir
    
    """Main function to run the pipeline"""
    # Pipeline
    logging.info("Loading data...")
    abundance, labels = load_data(abundance_file=abundance_path, metadata_file=metadata_path, target_column=target_column)
    logging.info(f"Data loaded: {abundance.shape[0]} features, {abundance.shape[1]} samples")

    logging.info("Filtering and transforming data...")
    filtered_data = filter_data(abundance)
    clr_data = clr_transform(filtered_data)

    logging.info("Training the model...")
    model = train_model(clr_data, labels)

    logging.info("Predicting probabilities...")
    probs = model.predict_proba(clr_data)[:, 1]
    # save the results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        probs_df = pd.DataFrame(probs, index=clr_data.index, columns=['Probabilities'])
        probs_df.to_csv(os.path.join(output_dir, 'predicted_probabilities.csv'), index=True)
        logging.info(f"Predicted probabilities saved to {os.path.join(output_dir, 'predicted_probabilities.csv')}")

    logging.info("Plotting ROC curve...")
    plot_roc(labels, probs, save_path=output_dir, show=False)

    logging.info("Pipeline completed successfully.")
#%%
if __name__ == "__main__":
    main()

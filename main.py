import numpy as np

rf_model = None

data = np.random.randint(0, 100, 100)

abundance_cutoff = 0.00001
prevalence_cutoff = 0.05

mean_abundance = None  # TODO

# def clr_transform(data, log_n0: float = 1e-06):
#     pass

# def predict(model, data, type):
#     return data

#%%

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#%%
def load_data(abundance_path, metadata_path):
    """Load and preprocess data"""
    abundance = pd.read_csv(abundance_path, sep='\t', index_col='Species')
    metadata = pd.read_csv(metadata_path, sep='\t')
    
    # Clean metadata and get labels
    metadata_clean = metadata.dropna(subset=['Group'])
    sample_labels = metadata_clean['Group'].astype('category').cat.codes
    
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

def train_model(X, y):
    """Train Random Forest classifier with cross-validation"""
    param_grid = {'max_features': np.unique(np.linspace(1, X.shape[1], 5, dtype=int))}
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search

def plot_roc(y_true, y_probs):
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
    plt.show()


#%%
def main():

    ABUNDANCE_PATH = "/Users/lawrenceadu-gyamfi/Documents/PERSONAL/PROJECTS/ML_Microbiome_Package/Dataset/abundance_crc.txt"
    METADATA_PATH = "/Users/lawrenceadu-gyamfi/Documents/PERSONAL/PROJECTS/ML_Microbiome_Package/Dataset/metadata_crc.txt"

    # Pipeline
    abundance, labels = load_data(ABUNDANCE_PATH, METADATA_PATH)
    filtered_data = filter_data(abundance)
    clr_data = clr_transform(filtered_data)

    model = train_model(clr_data, labels)
    probs = model.predict_proba(clr_data)[:, 1]

    plot_roc(labels, probs)

#%%
if __name__ == "__main__":
    main()
# %%

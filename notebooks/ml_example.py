# %%
from sklearn.model_selection import train_test_split
from microfactual.data_processing import load_data, filter_data, clr_transform
from microfactual.modeling import train_model
from microfactual.visualisation import plot_roc


# %%
abundance_path = (
    "../NSCLC_Dataset/Clean_Dataset/abundance_nsclc_discovery_renamed_clean_relab.tsv"
)
metadata_path = (
    "../NSCLC_Dataset/Clean_Dataset/metadata_nsclc_discovery_renamed_clean.tsv"
)
target_column = "OS12"
sample_column = "Run"
abundance, labels = load_data(
    abundance_file=abundance_path,
    metadata_file=metadata_path,
    target_column=target_column,
    sample_column=sample_column,
)
filtered_data = filter_data(abundance)
clr_data = clr_transform(filtered_data)


# %%
# split data
train_x, test_x, train_y, test_y = train_test_split(
    clr_data, labels, test_size=0.2, random_state=42, stratify=labels
)
# %%
model = train_model(train_x, train_y, n_jobs=3, cv_splits=10, n_estimators=200)
# %%
test_probs = model.predict_proba(test_x)[:, 1]
plot_roc(test_y, test_probs)

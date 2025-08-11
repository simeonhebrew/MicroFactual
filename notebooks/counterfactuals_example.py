# %%
import dice_ml
from sklearn.model_selection import train_test_split
from microfactual.data_processing import load_data, filter_data, clr_transform
from microfactual.modeling import train_model

# %%
abundance_path = "../Dataset/abundance_crc.txt"
metadata_path = "../Dataset/metadata_crc.txt"
target_column = "Group"
abundance, labels = load_data(
    abundance_file=abundance_path,
    metadata_file=metadata_path,
    target_column=target_column,
)
filtered_data = filter_data(abundance)
clr_data = clr_transform(filtered_data)


# %%

model = train_model(clr_data, labels, n_jobs=1)
# %%
clr_data['target'] = labels.values
dataset = clr_data
target = dataset['target']
train_dataset, test_dataset, _, _ = train_test_split(
    dataset, target, test_size=0.2, random_state=0, stratify=target
)
# model = LogisticRegression()
# model.fit(train_dataset.drop(columns="target"), train_dataset["target"])
# Dataset for training an ML model
d = dice_ml.Data(
    dataframe=train_dataset,
    continuous_features=dataset.columns[:-1].tolist(),
    outcome_name='target',
)

# Pre-trained ML model
m = dice_ml.Model(model=model, backend='sklearn')
# DiCE explanation instance
exp = dice_ml.Dice(d, m, method="genetic")
# %%
# Generate counterfactual examples
queries = test_dataset[test_dataset["target"] == 1].drop(columns="target")
query_instance = queries[0:1]
dice_exp = exp.generate_counterfactuals(
    query_instance, total_CFs=5, desired_class="opposite", verbose=True
)
# Visualize counterfactual explanation
dice_exp.visualize_as_dataframe(show_only_changes=True)
# %%

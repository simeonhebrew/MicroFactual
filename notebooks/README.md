# Notebooks

A small, curated set of runnable examples on the shipped colorectal-cancer
dataset (`datasets/abundance_crc.txt`, `datasets/metadata_crc.txt`). Run them
from this directory.

| Notebook | What it covers |
|----------|----------------|
| [`00_End_to_End_Feature_Tour.ipynb`](00_End_to_End_Feature_Tour.ipynb) | The full tour: dataset loading, `mf.explore` cutoff diagnostics, preprocessing, sklearn interop, and the headline **counterfactual** workflow (sparse/validated explanations, plausibility bounds, class-reference heatmap, cohort importance). |
| [`01_Quickstart_Classification.ipynb`](01_Quickstart_Classification.ipynb) | The one-liner `mf.classify()` path, with ROC and feature-importance plots. |
| [`02_Modular_Pipeline_Builders.ipynb`](02_Modular_Pipeline_Builders.ipynb) | Building custom sklearn preprocessing pipelines with `MicrobiomeClassifier`. |

Counterfactual explanations are MicroFactual's headline capability — start with
notebook `00`.

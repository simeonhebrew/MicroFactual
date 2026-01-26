# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-01-26

### Fixed
- **Sklearn Compatibility**: Fixed `ValueError` in `MicrobiomeClassifier` by correcting inheritance order (MRO) with `ClassifierMixin` and `BaseEstimator`.
- **Dashboard Feature Names**: Fixed `ValueError` in `RandomForestClassifier` when launching dashboard, by automatically converting DataFrame inputs to numpy arrays in `predict` methods to bypass strict feature name checks (caused by dashboard sanitization).
- **Dashboard Indexing**: Fixed `IndexingError` in `AbundanceFilter` and `PrevalenceFilter` when column names are renamed (e.g., by dashboard), by using robust boolean masking.
- **Notebooks**: Patched `01_Quickstart` and `03_Dashboard` to use `get_feature_names_out()` for feature importance plotting.
- **Transforms**: Added `get_feature_names_out()` to `AbundanceFilter`, `PrevalenceFilter`, and `CLRTransform` for pipeline compatibility.


## [0.2.0] - 2025-06-25

Major architectural overhaul introducing a modular, sklearn-compatible design.

### Added

#### Core Architecture
- `MicrobiomeDataset`: Central data container with `X`, `y` properties and provenance tracking.
- `MicrobiomeClassifier`: "Batteries-included" classifier wrapper with built-in preprocessing.
- `mf.classify()`: High-level one-liner API for quick classification workflows.

#### Visualization Module (`microfactual.visualization`)
- `plot_roc()`: Generate ROC curves with AUC scores.
- `plot_confusion_matrix()`: Generate confusion matrices with custom class labels.
- `plot_feature_importance()`: Horizontal bar charts for feature importance.
- `launch_dashboard()`: Helper to launch interactive ExplainerDashboard.
- All visualization functions return `matplotlib.Figure` objects for flexibility.

#### Explainability Module (`microfactual.explainability`)
- `BaseExplainer`: Abstract base class for decoupling explainability frameworks.
- `DiCEExplainer`: Adapter for [DiCE](https://github.com/interpretml/DiCE) to generate counterfactual explanations.

#### Preprocessing (`microfactual.preprocessing`)
- `AbundanceFilter`: Sklearn-compatible transformer for abundance filtering.
- `PrevalenceFilter`: Sklearn-compatible transformer for prevalence filtering.
- `CLRTransform`: Sklearn-compatible transformer for Centered Log-Ratio transformation.

### Deprecated
- `run_pipeline()`: Deprecated in favor of `mf.classify()`. Will be removed in v1.0.

### Changed
- Refactored project structure to separate concerns (core, models, preprocessing, visualization, explainability).
- Updated documentation with new API references and usage examples.

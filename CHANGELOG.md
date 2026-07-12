# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Lean core install**: `dice-ml` and `explainerdashboard` moved out of core dependencies into an optional `explainability` extra. Install with `pip install 'microfactual[explainability]'`. Counterfactual and dashboard entry points raise a clear ImportError pointing to the extra when it isn't installed.
- Moved `ruff` from runtime dependencies to the dev dependency group (it is a lint tool, not a runtime requirement).
- **Narrative reframing**: repositioned MicroFactual around interpretable, sklearn-native counterfactual explanations for microbiome classification. Updated package `description`, keywords, and README headline accordingly.
- Rewrote the README roadmap (removed duplicated entries) around the v0.2.0 release plan.

### Added
- **`mf.explain_counterfactual()`**: first-class, documented one-call entry point for per-sample counterfactual explanations (wraps `DiCEExplainer`). Exported at the top level.
- **Actionable counterfactuals**: `explain_counterfactual()` now applies post-hoc sparsification (`sparse=True`) — greedily reducing each counterfactual to a near-minimal set of taxa changes that still flips the prediction (on the CRC model this cut changes from 220/220 taxa to ~10). Added `features_to_vary` and `permitted_range` to constrain the search to modifiable taxa / plausible ranges.
- **`CounterfactualResult`**: interpretable wrapper returned by `explain_counterfactual()` with `.changes()` (taxon, original→counterfactual, delta, direction), `.n_changes`, `.validity` (fraction that truly flips), and `.summary()`. Pass `return_raw=True` for the underlying DiCE object.
- **`mf.counterfactual_importance()`**: cohort-level aggregation that runs counterfactuals across many samples and ranks taxa by how often they must change to flip predictions, with a net direction — a local-aggregated importance complementing global feature importance.
- Counterfactuals methodology documentation page (`docs/counterfactuals.rst`) covering assumptions, interpretation guidance, and limitations.
- `CITATION.cff` for GitHub "Cite this repository" support (paper DOI placeholder pending).
- End-to-end feature-tour notebook (`notebooks/00_End_to_End_Feature_Tour.ipynb`) exercising the full public API on the shipped Zeller 2014 CRC dataset.

### Fixed
- Version metadata is now consistent: `pyproject.toml` bumped to `0.2.0` to match `microfactual.__version__`.
- `MicrobiomeClassifier` now exposes forwarded model kwargs (e.g. `n_estimators`, `max_depth`) through `get_params`/`set_params`, so `sklearn.clone`, `set_params`, and `GridSearchCV` can tune the underlying classifier's hyperparameters. Previously these raised `Invalid parameter` under GridSearchCV.
- CI lint (`ruff check src/microfactual`) now passes: migrated ruff config to the `[tool.ruff.lint]` section, ignored ML naming conventions (`X`/`y`) and undocumented `__init__`, per-file-ignored `E402` in the deprecated shim modules, and documented the `y` parameter on transformer `fit` methods.

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

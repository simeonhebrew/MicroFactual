# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- README now leads with a runnable counterfactual example (the headline feature) instead of `mf.classify()`; features list, architecture diagram, and API reference updated to cover the counterfactual, plausibility, and data-exploration APIs.
- **Lean core install**: `dice-ml` moved out of core dependencies into an optional `explainability` extra. Install with `pip install 'microfactual[explainability]'`. The counterfactual entry points raise a clear ImportError pointing to the extra when it isn't installed.
- Moved `ruff` from runtime dependencies to the dev dependency group (it is a lint tool, not a runtime requirement).
- **Narrative reframing**: repositioned MicroFactual around interpretable, sklearn-native counterfactual explanations for microbiome classification. Updated package `description`, keywords, and README headline accordingly.
- Rewrote the README roadmap (removed duplicated entries) around the v0.2.0 release plan.

### Added
- **PyPI release setup**: a `release.yml` workflow that publishes via PyPI Trusted Publishing (OIDC, no tokens) — TestPyPI on a `v*` tag, PyPI on a published GitHub Release. Added `[project.urls]`, single-source dynamic version (read from `microfactual.__version__`), a lean sdist (excludes example datasets/notebooks, 2.4 MB → 36 KB), and `RELEASING.md` with the setup + release process.
- **`mf.explore(dataset)`**: data-exploration panel for choosing preprocessing cutoffs — log-abundance histogram, prevalence histogram, and a joint prevalence-vs-abundance scatter, all overlaying the proposed `AbundanceFilter`/`PrevalenceFilter` cutoffs and reporting how many taxa are retained. Individual plotters (`plot_abundance_histogram`, `plot_prevalence_histogram`, `plot_prevalence_abundance`) are also exported.
- **`mf.explain_counterfactual()`**: first-class, documented one-call entry point for per-sample counterfactual explanations (wraps `DiCEExplainer`). Exported at the top level.
- **Actionable counterfactuals**: `explain_counterfactual()` now applies post-hoc sparsification (`sparse=True`) — greedily reducing each counterfactual to a near-minimal set of taxa changes that still flips the prediction (on the CRC model this cut changes from 220/220 taxa to ~10). Added `features_to_vary` and `permitted_range` to constrain the search to modifiable taxa / plausible ranges.
- **`CounterfactualResult`**: interpretable wrapper returned by `explain_counterfactual()` with `.changes()` (taxon, original→counterfactual, delta, direction), `.n_changes`, `.validity` (fraction that truly flips), and `.summary()`. Pass `return_raw=True` for the underlying DiCE object.
- **`mf.counterfactual_importance()`**: cohort-level aggregation that runs counterfactuals across many samples and ranks taxa by how often they must change to flip predictions, with a net direction — a local-aggregated importance complementing global feature importance.
- **Counterfactual plausibility & visualization**: `mf.plausible_range()` derives per-taxon bounds from a reference class (feed to `permitted_range` to keep counterfactuals in-distribution — cuts overshoot); `mf.counterfactual_concordance()` scores how well a counterfactual moves toward the reference-class median; and `mf.plot_counterfactual_heatmap()` visualizes a counterfactual against the class medians in reference-SD units.
- Counterfactuals methodology documentation page (`docs/counterfactuals.rst`) covering assumptions, interpretation guidance, and limitations.
- **Documentation site** published to GitHub Pages (<https://simeonhebrew.github.io/MicroFactual/>): rewrote the landing page around the counterfactual-first API, added a preprocessing-rationale guide (`docs/preprocessing.rst`), and added a docs badge + link to the README. The Sphinx build is now warning-free.
- `CITATION.cff` for GitHub "Cite this repository" support (paper DOI placeholder pending).
- End-to-end feature-tour notebook (`notebooks/00_End_to_End_Feature_Tour.ipynb`) exercising the full public API on the shipped Zeller 2014 CRC dataset.

### Removed
- **Retired the ExplainerDashboard integration**: removed `mf.launch_dashboard()` and the `microfactual.visualization.dashboard` module, and dropped `explainerdashboard` from the `explainability` extra (now just `dice-ml`). The dashboard was off-narrative for a counterfactual-first library; use the built-in plots and counterfactual explanations instead.
- Pruned the `notebooks/` folder to a curated, verified set (`00` feature tour, `01` classify quickstart, `02` modular pipelines) with an index `README`. Removed notebooks/scripts that used deprecated APIs or were superseded by the feature tour (`03_Interactive_Dashboard`, `04_Explainability_Counterfactuals`, `counterfactuals_example.ipynb`/`.py`, `explainerboard_visualisation.ipynb`, `ml_example.py`) and the committed `dice_cf_outputs/` artifacts.
- Removed the unused `BaseModel` abstract base class (it was exported but never implemented — `MicrobiomeClassifier` follows sklearn's base classes) and the orphaned pre-rename `src/microbiome_ml` package artifacts.

### Fixed
- Updated stale references to the old repository name `ML_Microbiome_Package` (README CI badge and citation, `CITATION.cff`, `Makefile`) to the current `MicroFactual` repo.
- **Clean import**: `import microfactual` no longer emits `DeprecationWarning`s. The functional implementations moved to non-deprecated modules (`core.processing`, `models.training`, `visualization.roc_io`); the deprecated `data_processing` / `modeling` / `visualisation` modules are now thin re-export shims that warn **only when imported directly**.
- **Feature names (T2.3)**: `MicrobiomeClassifier.predict`/`predict_proba` no longer strip DataFrame column names, removing sklearn's "X does not have valid feature names" `UserWarning` and keeping downstream sklearn tooling working (test-suite warnings dropped from ~32 to 2).
- `explain_counterfactual()` discards DiCE rows that don't actually flip the prediction, so results never contain zero-change "counterfactuals" (previously the default could yield `validity < 100%` and an empty `changes(0)`).
- `explain_counterfactual()` now always returns a single `CounterfactualResult` for a single-row query, even when DiCE finds no counterfactual (previously it returned a bare `[]`, causing `AttributeError: 'list' object has no attribute 'changes'` downstream). `CounterfactualResult.changes()` also handles zero counterfactuals without raising. Empty results report `n_counterfactuals == 0` / `validity == 0` and a "No counterfactuals found." summary.
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

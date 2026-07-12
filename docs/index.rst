MicroFactual
============

**Interpretable, sklearn-native counterfactual explanations for microbiome
classification.**

MicroFactual answers a question most microbiome ML tools can't: *"what minimal
change in taxa abundance would flip this sample's prediction?"* It pairs
per-sample counterfactual analysis with a clean scikit-learn-compatible surface
(``Pipeline``, ``GridSearchCV``, ``cross_val_score``) over microbiome-aware
preprocessing (abundance/prevalence filtering, CLR).

.. image:: https://github.com/simeonhebrew/MicroFactual/actions/workflows/ci.yml/badge.svg?branch=main
   :alt: CI
   :target: https://github.com/simeonhebrew/MicroFactual/actions/workflows/ci.yml

**Non-goals:** not a replacement for QIIME2's bioinformatics pipeline, not a
feature-engineering toolkit, not a diversity/phylogenetics library.

Pipeline overview
-----------------

.. mermaid::

    flowchart LR
        A[Feature table + metadata] -->|MicrobiomeDataset| B(Preprocessing)
        B -->|Abundance / Prevalence / CLR| C[MicrobiomeClassifier]
        C -->|explain_counterfactual| D[Counterfactuals]
        C -->|plots| E[ROC / importance]
        D -->|plausible_range + heatmap| F[Actionable, plausible explanations]

Installation
------------

Install from source (not yet published to PyPI):

.. code-block:: bash

    # Core install (lean)
    pip install -e .

    # With the counterfactual explainability stack (DiCE)
    pip install -e '.[explainability]'

Quickstart: a counterfactual
----------------------------

.. code-block:: python

    import microfactual as mf
    from microfactual import AbundanceFilter, PrevalenceFilter, CLRTransform

    # 1. Load a real feature table + metadata
    ds = mf.MicrobiomeDataset.from_files(
        "abundance.tsv", "metadata.tsv",
        target_column="Group", sample_column="Sample ID",
    )

    # 2. Preprocess into CLR space (real taxon names are preserved end-to-end)
    X = CLRTransform().fit_transform(
        PrevalenceFilter(min_prevalence=0.1).fit_transform(
            AbundanceFilter(min_abundance=1e-5).fit_transform(ds.X)))
    y = ds.y

    # 3. Fit an sklearn-compatible classifier in that space
    model = mf.MicrobiomeClassifier(preprocessing=None).fit(X, y)

    # 4. What minimal change flips this sample's prediction?
    cf = mf.explain_counterfactual(
        model, X.iloc[[0]], background_data=X, y=y,
        class_names=list(ds.target_names),
    )
    print(cf.summary())
    cf.changes(0)   # taxon, original -> counterfactual, delta, direction

See the :doc:`counterfactuals` guide for plausibility bounds, cohort-level
importance, and the class-reference heatmap, and :doc:`preprocessing` for how to
choose the filtering and transform defaults.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Guides

   counterfactuals
   preprocessing

.. toctree::
   :maxdepth: 2
   :caption: Reference

   modules

Links
-----

- Source: https://github.com/simeonhebrew/MicroFactual
- Runnable examples: ``notebooks/`` in the repository (start with the
  end-to-end feature tour).

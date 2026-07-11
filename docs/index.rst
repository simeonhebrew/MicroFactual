Microfactual Documentation
===========================

A user-friendly Python framework for microbiome machine learning workflows.

**Key Features:**
- Easy data loading and preprocessing for microbiome datasets
- Robust filtering and CLR transformation utilities
- Random Forest modeling with cross-validation and hyperparameter tuning
- Publication-ready ROC curve visualization
- Command-line interface for reproducible pipelines
- Type hints and numpy-style docstrings throughout

.. image:: https://img.shields.io/github/actions/workflow/status/<your-username>/<your-repo>/ci.yml?branch=main
   :alt: Build Status
   :target: https://github.com/<your-username>/<your-repo>/actions

Pipeline Overview
-----------------

.. mermaid::

    flowchart TD
        A[Input Data] -->|Load| B(Dataset)
        B -->|Filter| C[Preprocessing]
        C -->|Transform| D[CLR/Scaling]
        D -->|Train| E[Random Forest]
        E -->|Explain| F[Counterfactuals]
        E -->|Visualize| G[ROC Curves / Feature Importance]

Quickstart
----------

.. code-block:: bash

    pip install microfactual

    # Run the pipeline via CLI
    microfactual --abundance data/abundance.txt --metadata data/metadata.txt --output_dir results/

Usage as a library:

.. code-block:: python

    from microfactual.core.dataset import load_dataset
    from microfactual.preprocessing import filter_features, clr_transformation
    from microfactual.models import RandomForestClassifier
    from microfactual.explainability import CounterfactualExplainer

    # Load and process
    data = load_dataset('abundance.txt', 'metadata.txt')
    filtered_data = filter_features(data)
    transformed_data = clr_transformation(filtered_data)

    # Train
    model = RandomForestClassifier()
    model.fit(transformed_data, labels)

    # Explain
    explainer = CounterfactualExplainer(model)
    explanation = explainer.explain(instance)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   counterfactuals
   modules

API Reference
-------------

- The API documentation is auto-generated from the source code docstrings.

Contributing
------------

- Contributions are welcome! See the repository for guidelines.
- For more, see the `README <../README.md>`_ or visit the `GitHub repo <https://github.com/<your-username>/<your-repo>>`_.

---

*This documentation is generated with Sphinx.*

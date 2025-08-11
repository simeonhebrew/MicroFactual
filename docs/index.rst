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

.. figure:: pipeline_overview.png
   :alt: Pipeline Overview
   :align: center

   *A typical workflow: data loading → filtering → CLR transform → modeling → results.*

Quickstart
----------

.. code-block:: bash

    pip install microfactual
    microfactual --abundance path/to/abundance.txt --metadata path/to/metadata.txt --output_dir results/

Or use as a library:

.. code-block:: python

    from microfactual.data_processing import load_data, filter_data, clr_transform
    from microfactual.modeling import train_model

    abundance, labels = load_data('abundance.txt', 'metadata.txt')
    filtered = filter_data(abundance)
    clr = clr_transform(filtered)
    model = train_model(clr, labels)

Contents
--------

.. toctree::
   :maxdepth=2
   :caption: Contents:

   user_guide
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

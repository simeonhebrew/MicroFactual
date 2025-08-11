User Guide
==========

This section provides practical guides and tips for using Microbiome-ML in your research.

Typical Workflow
----------------

1. **Prepare your data**: Format your abundance and metadata tables as described in the README.
2. **Run the CLI**: Use the command-line interface for a full pipeline run.
3. **Use as a library**: Import and use individual functions for custom workflows.

Example: Full Pipeline
----------------------

.. code-block:: bash

    microbiome-ml --abundance data/abundance.txt --metadata data/metadata.txt --output_dir results/

Example: Custom Python Workflow
-------------------------------

.. code-block:: python

    from microfactual.data_processing import load_data, filter_data, clr_transform
    from microfactual.modeling import train_model
    from microfactual.visualisation import plot_roc

    abundance, labels = load_data('abundance.txt', 'metadata.txt')
    filtered = filter_data(abundance)
    clr = clr_transform(filtered)
    model = train_model(clr, labels)
    probs = model.predict_proba(clr)[:, 1]
    plot_roc(labels, probs, show=True)

Tips & Best Practices
---------------------

- Always check your input data for missing values and correct formatting.
- Use the CLI for reproducibility; use the library for flexibility.
- See the API Reference for details on each function.

For more, see the `README <../README.md>`_.

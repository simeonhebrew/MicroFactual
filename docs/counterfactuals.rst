Counterfactual Explanations
===========================

Counterfactual explanations are MicroFactual's headline capability. For an
individual sample they answer:

    *"What is the smallest change in taxa abundance that would flip this
    sample's predicted class?"*

Unlike global feature-importance scores, a counterfactual is **per-sample and
actionable**: it names specific taxa and the direction/magnitude of change that
would move the sample across the decision boundary. This maps naturally onto
clinical and biological reasoning ("which microbes, and by how much?").

Quickstart
----------

.. code-block:: python

    import microfactual as mf

    # Fit a model in the feature space you want to explain (see note below).
    model = mf.MicrobiomeClassifier(preprocessing=None).fit(X_clr, y)

    # One call: what would flip the first sample's prediction?
    cf = mf.explain_counterfactual(
        model,
        query=X_clr.iloc[[0]],
        background_data=X_clr,
        y=y,
        total_CFs=3,
    )

    # Show only the features that changed
    cf.visualize_as_dataframe(show_only_changes=True)

``explain_counterfactual`` wraps :class:`~microfactual.explainability.counterfactuals.DiCEExplainer`
(built on `DiCE <https://github.com/interpretml/DiCE>`_). Use the class directly
when you need to reuse one explainer across many query samples.

Methodology & assumptions
--------------------------

- **Search space = model input space.** DiCE perturbs the *inputs* the model was
  trained on. Fit the model, and pass ``query``/``background_data``, in the
  **same feature space** — if the model was trained on CLR-transformed
  abundances, explain in CLR space. Explaining a model whose preprocessing is
  baked into the estimator (``preprocessing="auto"``) will search over the raw
  inputs, which is usually *not* what you want; transform first and use
  ``preprocessing=None``.
- **Background data defines plausibility.** DiCE samples candidate
  counterfactuals guided by ``background_data``. A representative training set
  yields more realistic counterfactuals; a small or skewed one yields less
  trustworthy ones.
- **Diversity vs. proximity.** ``total_CFs`` controls how many alternative
  counterfactuals are returned. Multiple counterfactuals expose *different*
  minimal routes across the boundary rather than a single point estimate.

Interpretation guidance
------------------------

- Read a counterfactual as *"a sample like this one but with these taxa shifted
  would likely be classified as the other group"* — a statement about the
  **model**, not a validated biological intervention.
- Prefer changes that are consistent across several counterfactuals and across
  several query samples of the same class; single-counterfactual changes can be
  artifacts of the search.
- Report the feature space (raw / relative-abundance / CLR) alongside any
  counterfactual, since the magnitude of a "minimal change" is only meaningful
  relative to that space.

Limitations
-----------

- **Compositional constraints are not enforced.** Counterfactuals in CLR space
  are not guaranteed to map back to a valid composition on the simplex (values
  summing to a constant). Treat the suggested shifts as directional rather than
  exact recipes. Constrained, simplex-aware counterfactuals are on the roadmap
  (see the release plan, T4.3).
- **Binary classification only** in the current release.
- **Correlated taxa.** Microbiome features are highly correlated; a
  counterfactual may shift one taxon as a proxy for a correlated group. Corroborate
  with domain knowledge before drawing biological conclusions.
- Requires the optional ``explainability`` extra
  (``pip install 'microfactual[explainability]'``).

See also the end-to-end notebook ``notebooks/00_End_to_End_Feature_Tour.ipynb``
for a runnable example on a real colorectal-cancer dataset.

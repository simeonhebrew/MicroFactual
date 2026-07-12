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

    print(cf.summary())
    # -> "3 counterfactual(s) flipping CRC -> Control; features changed:
    #     min=10, median=10, max=14; validity=100%."

    cf.changes(0)   # tidy table: feature, original, counterfactual, delta, direction

``explain_counterfactual`` returns a
:class:`~microfactual.explainability.result.CounterfactualResult` (pass
``return_raw=True`` for the underlying `DiCE <https://github.com/interpretml/DiCE>`_
object). It exposes ``.changes()``, ``.n_changes``, ``.validity`` and
``.summary()``.

Actionable (sparse) counterfactuals
-----------------------------------

By default DiCE's genetic search perturbs *every* feature, which is useless for
interpretation. ``explain_counterfactual`` therefore applies **post-hoc
sparsification** (``sparse=True``): changed taxa are reverted one at a time,
smallest change first, keeping a revert only while the prediction still flips.
On a real CRC model this reduces a counterfactual from *220 taxa changed* to
*~10* — a genuinely actionable "change these few taxa" statement.

Two further controls target actionability directly:

- ``features_to_vary=[...]`` restricts the search to taxa you consider
  modifiable (e.g. diet-associated genera), leaving host/fixed factors alone.
- ``permitted_range={feature: [lo, hi]}`` bounds counterfactual values to a
  plausible range.

Cohort-level counterfactual importance
--------------------------------------

:func:`~microfactual.explainability.counterfactuals.counterfactual_importance`
runs counterfactuals across many samples and ranks taxa by **how often they must
change to flip predictions**, with a net direction:

.. code-block:: python

    imp = mf.counterfactual_importance(model, X_clr, y, top_n=10)
    # columns: feature, n_samples, frequency, mean_delta, direction

Unlike a global feature-importance score, this is *local-aggregated*: it reflects
the taxa that actually drive individual decisions across the cohort.

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

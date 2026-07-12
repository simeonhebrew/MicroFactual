Preprocessing: choosing filters and transforms
===============================================

MicrobiomeClassifier's ``preprocessing="auto"`` applies three steps in order:

1. :class:`~microfactual.preprocessing.transforms.AbundanceFilter` — drop
   very-low-abundance taxa (``min_abundance=1e-6``).
2. :class:`~microfactual.preprocessing.transforms.PrevalenceFilter` — drop taxa
   present in too few samples (``min_prevalence=0.01``).
3. :class:`~microfactual.preprocessing.transforms.CLRTransform` — centered
   log-ratio transform (``pseudocount=1e-6``).

Those are the values ``"auto"`` uses. Note the standalone ``PrevalenceFilter``
class defaults to ``min_prevalence=0.05``; ``"auto"`` deliberately sets a looser
``0.01``. Use :func:`~microfactual.visualization.exploration.explore` to pick
these from your own data rather than trusting any default blindly.

Why filter at all?
------------------

Microbiome feature tables are wide and sparse (often 80–95% zeros). Most taxa are
low-abundance, rarely observed, or sequencing noise / contaminants. Filtering:

- removes features that carry little signal and inflate the multiple-comparison
  and overfitting burden;
- stabilises the CLR transform, whose geometric mean is sensitive to a long tail
  of near-zero values.

Abundance cutoff
----------------

The **log10 histogram of per-taxon mean abundance is typically bimodal** — a
low-abundance "noise" mode and a higher "signal" mode. The trough between them is
the natural cutoff. ``mf.explore(dataset)`` draws this histogram with the
proposed cutoff overlaid, so you can see whether ``1e-6`` sits in the trough for
*your* data (it may not — relative-abundance vs. count tables differ by orders of
magnitude).

- **Use a higher cutoff** when you only trust well-quantified taxa, or the table
  is relative abundance already normalised to 1.
- **Use a lower cutoff** for deeply-sequenced count data where rare taxa are
  still reliable.

Prevalence cutoff
-----------------

Prevalence is the fraction of samples in which a taxon appears. The distribution
is typically U-shaped (taxa present almost never or almost always). ``"auto"``
uses ``0.01`` (removes taxa seen in <1% of samples); the standalone class
defaults to ``0.05``. Raise it (e.g. ``0.1``) when you want features informative
across the cohort rather than a handful of samples — often the bigger lever on
model stability than the abundance cutoff.

CLR transform
-------------

Microbiome abundances are **compositional**: they are relative (constrained to a
constant sum), so only *ratios* between taxa carry information, not absolute
values. The centered log-ratio (CLR) maps compositions to a real-valued space
where standard classifiers and distances behave sensibly, by taking the log of
each part divided by the sample's geometric mean.

- The ``pseudocount`` replaces zeros before the log. Its value affects how
  strongly zeros are pulled toward the low end; report it alongside results.
- CLR keeps real taxon names, so counterfactuals and importances stay
  interpretable (see :doc:`counterfactuals`).

When *not* to CLR: if you plan to interpret absolute abundances directly, or your
downstream model already assumes compositional structure, CLR may not be
appropriate.

Known limitations
-----------------

- **Zero handling is pseudocount-based**, not model-based; results can be
  sensitive to the pseudocount on very sparse tables.
- **CLR only**, no ILR/ALR. CLR components are not fully independent (they sum to
  zero by construction). An isometric log-ratio (ILR) basis avoids this but needs
  a chosen partition; it is on the roadmap.
- Filtering thresholds are **global**, not per-class; extremely class-imbalanced
  taxa may be dropped. Inspect with ``mf.explore`` before committing.

Doing it manually
-----------------

The transforms are plain scikit-learn transformers, so you can compose them
yourself (and preserve real taxon names):

.. code-block:: python

    from microfactual import AbundanceFilter, PrevalenceFilter, CLRTransform

    X = CLRTransform().fit_transform(
        PrevalenceFilter(min_prevalence=0.1).fit_transform(
            AbundanceFilter(min_abundance=1e-5).fit_transform(dataset.X)))

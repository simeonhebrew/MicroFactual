# MicroFactual — Release & Narrative Plan

Target: a focused, defensible v0.2.0 release aligned with Simeon's research paper. This document captures the **narrative reframing** and the **concrete change list** that the spec-driven work will execute against.

---

## 1. Narrative Reframing

### Current pitch (implicit)
"A user-friendly Python framework for microbiome machine learning workflows."
→ This invites direct comparison to QIIME2 / scikit-bio / calour / q2-mlab, where MicroFactual is necessarily thinner.

### New pitch
**"MicroFactual: interpretable, sklearn-native counterfactual explanations for microbiome classification."**

Supporting positioning:
- **Primary contribution**: per-sample counterfactual analysis ("what minimal change in taxa abundance would flip this prediction?") — clinically and biologically meaningful, rare in microbiome ML tooling.
- **Secondary contribution**: a clean sklearn-compatible surface (Pipeline, GridSearchCV, cross_val_score) over microbiome-aware preprocessing (abundance/prevalence filters, CLR), so practitioners aren't forced into bespoke pipelines.
- **Non-goals (state explicitly)**: not a replacement for QIIME2's bioinformatics pipeline, not a feature-engineering toolkit, not a diversity/phylogenetics library.

### Audience
1. Microbiome researchers who already have feature tables (OTU/ASV/genus) and want interpretable classification.
2. Clinicians/translational researchers who care about *why* a sample was predicted a certain way at the individual level.
3. ML practitioners exploring counterfactual reasoning on compositional data.

### What this implies for the codebase
- The counterfactual path is **first-class**: documented up front, demoed in the README's headline example, fully tested, and called out in the paper.
- General classification utilities are **supporting infrastructure**, not the headline.
- Compositional-data limitations (CLR-only, zero-handling) must be **named and justified** rather than hidden.

---

## 2. Change List (grouped by release blocker tier)

### Tier 1 — Must-fix before tag (release blockers)

| # | Item | Files | Rationale |
|---|------|-------|-----------|
| T1.1 | Version sync: bump `pyproject.toml` to `0.2.0` to match `__init__.py` | `pyproject.toml` | Inconsistency is a sloppy-release red flag. |
| T1.2 | Update `description` and README headline to the counterfactual-first narrative | `pyproject.toml`, `README.md` | Aligns metadata with new pitch. |
| T1.3 | Deploy Sphinx docs to GitHub Pages; add docs badge to README | `.github/workflows/docs.yml`, `README.md`, `docs/conf.py` | Workflow exists; need publish step + landing page. |
| T1.4 | README headline example must be a counterfactual, not generic `classify()` | `README.md`, `notebooks/counterfactuals_example.ipynb` | Pitch must match first code block users see. |
| T1.5 | Move `dice-ml` and `explainerdashboard` to optional extras `[explainability]`; keep core lean | `pyproject.toml`, `src/microfactual/__init__.py`, `src/microfactual/explainability/` | Heavy deps shouldn't be mandatory; protects install footprint. |
| T1.6 | Ship one real microbiome dataset end-to-end notebook (CRC or IBD; e.g., curatedMetagenomicData subset) | `notebooks/`, `datasets/`, `docs/` | Synthetic-only demos look weak in a paper companion repo. |

### Tier 2 — Strongly recommended before paper submission

| # | Item | Files | Rationale |
|---|------|-------|-----------|
| T2.1 | "Choosing preprocessing" docs section: justify defaults (abundance=1e-6, prevalence=0.01, CLR), name when each is/isn't appropriate, mention ILR/zero-handling as known limitations | `docs/preprocessing.md` (new) | Pre-empts the obvious reviewer pushback on compositional handling. |
| T2.2 | Add XGBoost (and one of LightGBM/SVM) classifier wrapper | `src/microfactual/models/` | "Only RF + LogReg" is too narrow for a published tool. |
| T2.3 | Fix sklearn feature-name handling — keep names through pipeline via `set_config(transform_output="pandas")` instead of silent numpy conversion | `src/microfactual/models/`, `src/microfactual/preprocessing/` | Current behavior masks warnings and breaks downstream sklearn tooling. |
| T2.4 | Counterfactual API: clear, documented entry point (`mf.explain_counterfactual(model, sample)` style); not just notebook code | `src/microfactual/explainability/`, `docs/counterfactuals.md` (new) | If counterfactuals are the headline, they need a first-class API. |
| T2.5 | Add a "Counterfactuals for microbiome" methodology section to docs (assumptions, limitations, interpretation) | `docs/counterfactuals.md` | Paper-grade rigour. |
| T2.6 | Real dataset benchmark: AUC / F1 vs. a baseline (QIIME2 classify-samples or sklearn-only) on the shipped dataset | `notebooks/benchmark.ipynb` (new) | Gives reviewers an evidence base. |

### Tier 3 — Quality / hygiene (post-release acceptable but worth doing)

| # | Item | Files | Rationale |
|---|------|-------|-----------|
| T3.1 | Enforce mypy in CI (`mypy src/microfactual --strict` with a small ignore list) | `.github/workflows/`, `mypy.ini` | Types exist but aren't enforced. |
| T3.2 | Finish `visualisation` / `visualization` deprecation — remove the legacy module in v1.0 with a clear timeline in CHANGELOG | `src/microfactual/visualisation.py`, `CHANGELOG.md` | Two spellings is confusing. |
| T3.3 | Remove duplicate "XGBoost, SVM" roadmap line; rewrite roadmap with realistic milestones | `README.md` | Copy-paste error reads as low-care. |
| T3.4 | Cite DiCE, ExplainerDashboard, scikit-learn, scikit-bio properly in README + paper | `README.md`, `CITATION.cff` (new) | Academic norm. |
| T3.5 | Add `CITATION.cff` with paper preprint/DOI placeholder | `CITATION.cff` (new) | Enables GitHub "Cite this repository" UI. |
| T3.6 | Add docstring coverage check (interrogate or pydocstyle) at >=80% on `src/` | CI config | Improves docs site quality. |
| T3.7 | Pin dev dependencies in `uv.lock`; document `make dev` / `uv sync --group dev` flow in CONTRIBUTING.md | `CONTRIBUTING.md` (new) | Helps external contributors. |

### Tier 4 — Stretch / future paper-worthy additions

| # | Item | Rationale |
|---|------|-----------|
| T4.1 | ILR transform alongside CLR | Addresses the compositional-rigor critique structurally. |
| T4.2 | Phylogeny-aware feature aggregation (genus → family roll-up) | Distinct from QIIME2's API; bridges to biology. |
| T4.3 | Counterfactual constraints (e.g., bound to plausible compositional shifts, enforce simplex constraint) | A genuine research contribution on top of DiCE. |
| T4.4 | Multi-class and regression support (currently binary classification only) | Broadens applicability. |

---

## 3. Suggested Spec Sequence

For the spec-driven phase, I'd propose specs in this order (each small enough to ship and review independently):

1. **spec-001-version-and-metadata** — T1.1, T1.2, T3.3, T3.5 (lowest risk, unblocks tagging).
2. **spec-002-optional-extras** — T1.5 (refactor imports so DiCE/ExplainerDashboard load lazily; touches `__init__.py` and `explainability/`).
3. **spec-003-counterfactual-api** — T2.4, T2.5 (the headline feature gets a real API + docs).
4. **spec-004-docs-deployment** — T1.3, T2.1, T3.6 (Sphinx → Pages, preprocessing rationale page).
5. **spec-005-real-dataset-and-readme** — T1.4, T1.6, T2.6 (shipped dataset, benchmark notebook, README rewrite around counterfactuals).
6. **spec-006-feature-names-and-models** — T2.3, T2.2 (sklearn correctness + XGBoost).
7. **spec-007-ci-hardening** — T3.1, T3.2, T3.7 (mypy, deprecation cleanup, contributor docs).
8. **spec-008-compositional-depth** *(post-release)* — T4.1, T4.3 (ILR + constrained counterfactuals; potential follow-up paper material).

---

## 4. Definition of "Release-Ready"

A v0.2.0 release is ready when **all of Tier 1 and Tier 2** are complete, specifically:

- [ ] `pip install microfactual` installs a small core; `pip install microfactual[explainability]` adds DiCE/ExplainerDashboard.
- [ ] README opens with a counterfactual example on a real dataset.
- [ ] `https://<org>.github.io/microfactual/` serves built docs with: quickstart, preprocessing rationale, counterfactuals methodology, API reference.
- [ ] One shipped notebook reproduces an end-to-end CRC/IBD classification + counterfactual analysis.
- [ ] One benchmark notebook compares AUC/F1 against a non-MicroFactual baseline.
- [ ] All 48+ tests pass; CI green; version metadata consistent.
- [ ] `CITATION.cff` present with the paper reference (preprint DOI acceptable).

---

## 5. Open Questions for Simeon

Before starting the specs, confirm:

1. **Which dataset** to ship as the canonical example — CRC (Zeller et al.), IBD (HMP2), or something tied directly to the paper?
2. **Paper status & DOI**: is there a preprint we can cite now, or do we ship with a "to appear" placeholder?
3. **License of any included dataset** — must be redistributable, or we ship a download script instead of raw data.
4. **Is multi-class needed for v0.2.0**, or can it wait for v0.3.0?
5. **Repo / org name for docs hosting** — will it stay personal, or move to a research-group org before release?

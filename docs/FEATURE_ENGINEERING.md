# Feature Engineering Approach

## Label

From `claim_use_case_dataset.xlsx`, **`status`** is mapped to **`approved`**:

- **`Completed`** → positive class (`1`)
- **`Declined`** → negative (`0`)

---

## What the Model Sees

The classifier is a **`RandomForestClassifier`** inside a sklearn **`Pipeline`**:

### 1. Numeric block (median impute → `StandardScaler`)

| Column(s) | Notes |
|---|---|
| `excessFee`, `rrp`, `balanceRRP`, `oldBalanceRRP`, `deviceCost` | Raw money values |
| `*_log1p` variants of the above | Skew control via `np.log1p` |
| `rrp_minus_balance`, `fee_to_rrp` | EDA-aligned ratios from Notebook 01 (`fee_to_rrp` is `NaN` when `rrp` ≤ 0) |
| `policy_length_days`, `purchase_to_end_days` | Date-derived deltas |
| `issue_desc_len`, `product_name_len`, `product_desc_len` | Text length signals |
| `symptom_count` | Sum of twelve damage / component flags |
| `turnOnOff`, `touchScreen`, `smashed`, `frontCamera`, `backCamera`, `frontOrBackCamera`, `audio`, `mic`, `buttons`, `connection`, `charging`, `other` | Damage / component flags (as floats) |

### 2. Categorical block (`OneHotEncoder`, `infrequent_if_exist`, `max_categories=50`)

`coverage`, **`coverage_duration_tier`** (6 m / 12 m / 24 m–style bands from policy span days, same breakpoints as Notebook 01), `policyStatus`, `claimType`, `channel`, `country`, `retailerName`, `deviceType`, `productCode`, `model`, **`productName`**, **`persona`** (cleaned strings; optionally rule-based fallback in `engineer_features`).

Raw long text **`issueDesc`** / **`productDesc`** are dropped after their lengths are computed — the tree reads only **length proxies** (+ an optional separate LLM pipeline).

---

## What EDA Showed (Notebook `01_load_and_explore_data.ipynb`)

The notebook performed chi-square tests, Pearson correlations, pairwise z-tests with Holm–Bonferroni correction, and keyword heuristics across all feature candidates. Key findings are below.

### Money columns

| Feature | Finding |
|---|---|
| All money fields vs `approved` | Pearson `\|r\| < 0.07` — no strong **linear** relationship |
| `deviceCost` | Zero variance (always 0); dropped as informative predictor — only `rrp` is used as device-value proxy |
| `rrp` (binned, q=5) | **p ≈ 5.65e-9, Cramér's V ≈ 0.124** — moderate association; lowest price band (49–1 319 SEK) has the lowest approval rate (~77 %) |
| `excessFee` tier | Notable variation: fee = 35 → ~64 % approval; fee = 619 → ~90 %; **predictive** |
| `rrp_minus_balance` | ~6.3 % of rows show prior coverage erosion; weak marginal Pearson (r ≈ 0.024). **Now engineered explicitly** (alongside raw `rrp` / `balanceRRP`) |
| `fee_to_rrp` | Deductible-to–list-price ratio studied in Notebook 01; engineered as `excessFee / rrp` (undefined when `rrp` is 0) |

### Policy duration

| Feature | Finding |
|---|---|
| Exact `coverage_duration_days` | p ≈ 0.126 > 0.05 — **not significant** at exact day level (also min_exp < 5) |
| `coverage_duration_tier` (6 m / 12 m / 24 m) | **p ≈ 0.005, Cramér's V ≈ 0.066** — weak but significant; 6-month tier has the highest approval (~90 %), 12-month tier the lowest (~83 %) |
| Most claims | 12-month policies (~71 % of data, approval ≈ 83–90 % across tiers) |

### Categorical features

| Feature | Finding |
|---|---|
| `channel` | p ≈ 0.73, Cramér's V ≈ 0.03 — **not significant**; included for completeness, contributes minimal signal |
| `claimType` | p ≈ 0.30 — **not significant**; no pairwise comparison significant after Holm correction |
| `coverage` (ADLD vs ADLD+THEFT) | p ≈ 0.65 — **not significant** |
| `retailerName` | p ≈ 0.148 — **not significant** at the global level; included for potential latent effects |
| `model` (top-20 + Other) | **p ≈ 1.76e-6, Cramér's V ≈ 0.149** — moderate association; some device models have markedly different approval rates |

### Symptom flags

| Feature | Finding |
|---|---|
| `touchScreen` | **p ≈ 7.6e-7** — highly significant; touchscreen damage is the strongest single flag predictor |
| `audio` | **p ≈ 0.006** — significant |
| Other flags (`turnOnOff`, `smashed`, `frontCamera`, etc.) | Not significant individually; included as a flag bundle because they contribute in aggregate |
| Missing flag rows | Different approval base rate than flag-present rows — missing ≠ "not damaged"; imputed as `NaN` rather than 0 |

### `issueDesc` — translation & text use

- **2 884 unique descriptions** were translated to English via Gemini (`gemini-2.5-flash-lite`) and cached to `notebooks/issue_desc_en_cache.json`.
- The majority of entries are originally in Swedish (largely: phone drop / screen crack narratives); English entries were passed through unchanged.
- The **raw wording is not fed to the tree**. `issue_desc_len` is character count on **`issueDesc_en`** when present, else fallback columns; light stripping only (no LLM-free mojibake pass on that path).
- Rationale: raw text would require TF-IDF or embedding, inflating dimensionality significantly. The length signal is a cheap, stable proxy for narrative completeness, which correlates weakly with claim quality.

---

## Deterministic Text Cleaning (`app.ml.text_clean`)

Before computing lengths and the `productName` categorical:

1. Reverse common **Latin-1 mojibake** where UTF-8 bytes were mis-decoded (`vÃ¤` → `vä`), up to two passes.
2. Optional **`ftfy`** for harder mixed-encoding cases (`pip install ftfy`, already in requirements).
3. **Unicode NFKC** normalization + whitespace collapse.

This fixes many "garbage" characters **without** an LLM and keeps train/serve behaviour aligned.

---

## Why Random Forest? (Model Selection Rationale)

The EDA findings directly motivated the choice of **`RandomForestClassifier`** and ruled out simpler or more complex alternatives:

| Model considered | Why not used |
|---|---|
| **Logistic Regression** | All money columns have `\|r\| < 0.07` against the binary label, indicating non-linear structure. Significant variables (`rrp`, `model`, `touchScreen`) show bin-level effects that logistic regression without manual interaction terms would underfit. |
| **Gradient Boosting (XGBoost / LightGBM)** | A natural upgrade, noted as a future direction. Not used in V1 because: (a) Random Forest is simpler to debug, (b) the dataset (~2 880 rows) is small enough that variance from boosting is not clearly beneficial, (c) SHAP integration (needed for explainability) is equally straightforward with either. |
| **Neural Network / Deep Learning** | Dataset is too small (~2 880 rows, ~20 effective features after OHE) for stable deep models. No raw text embeddings are fed in, removing the main motivation for a neural backbone. |
| **Naive Bayes** | Requires feature independence; symptom flags are correlated (a smashed phone often also has touchscreen damage) — assumption violated. |
| **SVM** | Non-linear kernel SVM would work but is slow to tune, does not give probabilities natively, and offers no interpretability via feature importance. |

**Random Forest** was selected because:
- It handles mixed numeric + OHE categorical naturally without manual scaling (though `StandardScaler` is applied upstream for numeric stability).
- It is robust to the **weak individual signals** observed in EDA — bagged trees aggregate many weak effects.
- It handles **missing values** (via median imputation) and **high-cardinality categoricals** (`model` has 363 distinct values, collapsed to top-50 via OHE's `max_categories`).
- `class_weight="balanced_subsample"` accounts for the imbalanced outcome (~83 % approved) without oversampling.
- `RandomizedSearchCV` over `{n_estimators, max_depth, min_samples_leaf, max_features}` provides light hyper-parameter optimisation in 18 iterations with 4-fold CV, scored on **ROC-AUC** (more appropriate than accuracy for an imbalanced binary task).

---

## Exploratory Analysis (Notebooks)

[`notebooks/01_load_and_explore_data.ipynb`](../notebooks/01_load_and_explore_data.ipynb) cross-checks:

- **`rrp − balanceRRP`** divergence (prior coverage erosion)
- Pearson/chi-square correlations with `approved` for all money and date columns
- Channel/coverage/claimType splits with pairwise Holm-corrected z-tests
- **Keyword heuristics** on `issueDesc` translations (because there is **no** explicit "out of policy" column — only text + `status`)
- Symptom flag burden per retailer and model

**Notebook 01 vs serve:** Derived money fields and **`coverage_duration_tier`** mirror the EDA notebook; **`symptom_count`** is summed in serve for convenience (counts may duplicate what the RF can infer from the twelve flags alone).

---

## Smarter Upgrades (Ideas)

| Direction | Benefit |
|---|---|
| **TF-IDF / small embedding** on `issueDesc` (translated) alongside trees | Signals from narrative without huge dimensionality |
| **Target / cross-fold encoding** for high-cardinality `productCode` / `model` | Stable rare-level handling vs. large OHE |
| **XGBoost with monotonic constraints** if policy dictates directionality | Interpretable monotonic arrows on amounts |
| **Separate calibration** (`PlattScaling` / isotonic on holdout by region) | Business-ready probabilities |
| **SHAP TreeExplainer** per prediction | Explainability beyond global importances |
| **Re-train notebook** pinning `pandas` categorical dtypes identical to serve | Eliminate drift on string edge cases |
| **Ablation:** drop `fee_to_rrp` or tiers | Confirm marginal gain vs model complexity |

---

## LLM Repair Flow (Separate from Scoring)

Endpoint **`POST /v1/repair-text`** (`app/genai/text_repair.py`):

1. Run **deterministic** cleaning (same helpers as modelling).
2. If **`GEMINI_API_KEY`** is set, Gemini returns **JSON** with English-polished **`issueDesc` / `productName` / `productDesc`** (see `prompts/text_repair.txt`).

This does **not** change scores unless you merge repaired strings back into a claim and predict again — it exists for readability, downstream analytics, or human review.

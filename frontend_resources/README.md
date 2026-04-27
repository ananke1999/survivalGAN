# Data section — presentation assets

Self-contained handoff for the **"Data visualisation & cleaning"** part of the
SurvivalGAN-on-MSK-IMPACT presentation. Everything the front-end teammate
needs is already in this directory — no Python required.

```
data-section/
├── README.md                      ← you are here
├── figures/
│   ├── 01_dataset_positioning.svg / .png
│   ├── 02_cancer_pareto.html      (interactive)
│   ├── 02_cancer_pareto.png       (static fallback / preview)
│   └── 03_pipeline_funnel.svg / .png
├── tables/
│   ├── headline_numbers.md
│   └── headline_numbers.csv
├── data/
│   ├── dataset_positioning.csv    (raw numbers behind Figure 1)
│   ├── cancer_type_summary.csv    (raw numbers behind Figure 2)
│   └── pipeline_steps.csv         (raw numbers behind Figure 3)
└── notebooks/
    ├── 01_clean_and_visualize.py  (reproducibility — run only if numbers change)
    └── requirements.txt
```

---

## Suggested narrative arc for Step 1 (~3–5 minutes live)

1. **Where MSK-IMPACT sits** in the SurvivalGAN dataset landscape → **Figure 1**
2. **Cleaning pipeline** from raw registry to model-ready cohort → **Figure 3**
3. **What the cleaned cohort looks like by cancer type** (motivates RQ2) → **Figure 2**
4. **Headline numbers table** as the closing "what we feed Step 2" slide

Order on the page is at the front-end's discretion. Figure 1 → 3 → 2 is the
narrative I'd recommend; figure file names are just identifiers.

---

## Figure 1 — Dataset positioning

**Title**: *MSK-IMPACT vs. SurvivalGAN benchmark datasets*
**One-line caption**: Number of patients (log) vs. number of features (log)
for the five datasets used in Norcliffe et al. (2023) plus our cleaned
MSK-IMPACT cohort.
**Speaker note**: SEER is the only paper benchmark larger than MSK-IMPACT,
and it has only 6 features. PHEART is the closest in (size × features), but
PHEART is licensed and single-disease (heart failure); MSK-IMPACT is public
and pan-cancer.

**Files**: `figures/01_dataset_positioning.svg` (preferred) or `.png`
**Underlying data**: `data/dataset_positioning.csv`
**Embed**:
```html
<img src="figures/01_dataset_positioning.svg"
     alt="MSK-IMPACT vs. SurvivalGAN benchmark datasets"
     style="max-width:100%; height:auto;">
```

---

## Figure 2 — Cancer-type Pareto (interactive)

**Title**: *MSK-IMPACT cleaned cohort: top-15 cancer types + 'Other'*
**One-line caption**: Bars sorted by patient count; bar color encodes the
event rate (fraction deceased). Hover for n_patients, event rate, censoring
rate, and median follow-up months.
**Speaker note**: This is the chart that motivates RQ2. Pancreatic and
hepatobiliary cancers are yellow (high event rate, ~58–61%) while prostate
and endometrial are dark purple (~26–27%). A single SurvivalGAN cannot
reasonably model both regimes well — hence the subgroup analysis.

**Files**: `figures/02_cancer_pareto.html` (canonical, interactive)
**Static preview**: `figures/02_cancer_pareto.png` (if the site can't host
HTML iframes, this is the fallback — but you lose the tooltips)
**Underlying data**: `data/cancer_type_summary.csv`

**Embed (preferred — interactive)**:
```html
<iframe src="figures/02_cancer_pareto.html"
        title="Cancer-type Pareto"
        width="100%" height="600"
        style="border:0; display:block; margin:0 auto;">
</iframe>
```

The HTML loads Plotly from a CDN (`cdn.plot.ly/plotly-3.5.0.min.js`), so it
needs network access at view time. If the presentation will be shown
offline, swap to the PNG fallback.

**Embed (fallback — static)**:
```html
<img src="figures/02_cancer_pareto.png"
     alt="Cancer-type Pareto: top-15 cancer types + Other, color = event rate"
     style="max-width:100%; height:auto;">
```

---

## Figure 3 — Pipeline funnel

**Title**: *Cleaning pipeline: from raw registry to model-ready cohort*
**One-line caption**: Row counts at each gate of preprocessing, with the
final stratified 80/20 train-test split.
**Speaker note**: Patient-level dedup is the largest single drop (~12% of
rows) and is often skipped in oncology benchmarks — the same patient can
appear multiple times under different sample IDs. The 38.6% event rate is
high but expected for an aggressive-cancer cohort.

**Files**: `figures/03_pipeline_funnel.svg` (preferred) or `.png`
**Underlying data**: `data/pipeline_steps.csv`
**Embed**:
```html
<img src="figures/03_pipeline_funnel.svg"
     alt="Cleaning pipeline: from 54,331 raw rows to 34,938 train + 8,735 test"
     style="max-width:100%; height:auto;">
```

---

## Headline numbers table

A single small table that closes the section as a "this is what we feed
Step 2" reference card.

- Markdown source: `tables/headline_numbers.md` (drop straight into MD-aware site generators)
- CSV source: `tables/headline_numbers.csv` (build a custom HTML table in the front-end)

Suggested use: render right after Figure 3, with a section heading like
*"Final cohort, in numbers"*.

---

## Style guide for consistency with the site

If the front-end lets you set Plotly defaults globally, match these to keep
the interactive chart visually consistent with the rest of the page:

- **Primary blue**: `#1e3a8a` (used in funnel numbers, train branch border)
- **Accent orange**: `#ea580c` (used for MSK-IMPACT highlight, test branch border)
- **Muted text**: `#475569`
- **Body text**: `#0f172a`
- **Panel fill**: `#f1f5f9`
- **Border**: `#cbd5e1`
- **Font**: Inter / system-ui sans-serif (Plotly chart already targets this)

The static SVGs are exported with the same palette so they should sit next
to each other without clashing.

---

## Regenerating (only if upstream data changes)

The figures and CSVs are committed — you only need to regenerate if the
preprocessing rules change or the underlying CSV is updated.

```bash
cd notebooks/
pip install -r requirements.txt
python vis_preprocess.py --csv /path/to/msk_impact_50K.csv
```

The script writes back into `figures/`, `tables/`, and `data/` (the parent
of `notebooks/`). All randomness is seeded (`random_state=42`); re-runs
produce byte-identical outputs given the same input CSV.

The script needs Chrome only if you want to regenerate the static
`02_cancer_pareto.png`. Without Chrome, `kaleido` will silently skip and
the HTML version is still produced.

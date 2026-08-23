# Capstone Report — Machine Learning Lane

- **Author:** Shubham Sharma
- **Lane:** Machine Learning
- **Repo:** [https://github.com/ShubhamSnSharma/flyrank-ml-internship](https://github.com/ShubhamSnSharma/flyrank-ml-internship)
- **Date:** August 2026
- **Paper Live URL:** [https://shubhamsnsharma.github.io/flyrank-ml-internship/](https://shubhamsnsharma.github.io/flyrank-ml-internship/)

---

## Title

**Prioritizing Enterprise Content for Organic Search Traffic Decay Risk: A Machine Learning Approach to Decision Support**

## Abstract

1. **Question**: Can machine learning models help prioritize content pages vulnerable to future organic search traffic decay to guide scarce human editorial review bandwidth?
2. **Data & Method**: Across 79M+ rows of FlyRank enterprise search-performance data, we apply activity-based eligibility criteria to analyze 16,513 eligible pages across 36 client domains. We train a standardized Logistic Regression classifier on 9 pre-May historical modeling features under a client-grouped holdout validation design (`GroupShuffleSplit`, `test_size=0.20`, 8 held-out clients) to predict whether May organic search clicks decline by $\ge 20\%$ relative to April.
3. **Primary Result**: On the primary client-held-out test partition (Seed 42, $N=1,727$ pages), Logistic Regression achieved a Precision@50 of **0.2400** (12 true declining pages identified out of 50) compared to **0.2000** (10/50) for a transparent 3-signal heuristic baseline, representing a measured single-split lift of **+0.0400 (+4.0 percentage points)**.
4. **Validation Caveat**: An expanded 5-split stability audit showed that observed lift varies substantially across client partitions (mean lift: +1.6 percentage points, range: -12.0 to +14.0 pp), with Logistic Regression outperforming the baseline in 3 splits and underperforming in 2.
5. **Practical Use**: We translate predictions into a decision-support Content Action Playbook with 5 illustrative review archetypes, explainable reason codes, illustrative planning effort assumptions, and human-in-the-loop safety guardrails that prohibit autonomous CMS modifications.

---

## 1. Problem framing

### Decision Supported
In enterprise digital publishing, organizations manage content libraries spanning thousands of URLs. Organic search traffic can change over time, but editorial teams operate under finite diagnostic bandwidth and cannot manually investigate every page with equal depth. For this study, 15–20 high-depth page reviews per week are used as an illustrative editorial planning assumption.

This research supports **candidate queue prioritization for human editorial content review**. Rather than attempting to automate editorial interventions or reverse-engineer search engine ranking algorithms, the model ranks URLs by their predicted likelihood of meeting the study's subsequent click-decay label, allowing editorial teams to direct diagnostic attention toward pages showing empirical decline signals.

### Unit of Analysis & Target Decision
- **Unit of Analysis**: An individual published URL (`content_hash_id`) within a client domain (`client_hash_id`) over the May 1–31, 2026 forward evaluation window.
- **Output**: A deterministic, ranked decision-support candidate queue containing predicted decay probabilities, 5 review archetype tags, explainable reason codes, illustrative planning effort assumptions, and safety guardrails.
- **Human Action**: Content editors conduct structured diagnostic audits (intent verification, SERP competitor analysis, factual updates, and internal linking improvements) based on the assigned review archetype.
- **Target Decision**: Prioritizing pages for human diagnostic review, with the top 50 candidates used as the primary evaluation queue.

### Cost of Errors & Asymmetry
- **Cost of False Positives**: Allocates illustrative editorial review effort (~2.0–5.0 hours planning assumption per page) to a page that does not meet the study's May decline label.
- **Cost of False Negatives**: False negatives represent pages that may receive lower review priority even though they subsequently meet the study's decline label.
- **Operating Asymmetry**: Because human review bandwidth is strictly capped, ranking precision at the top of the queue (Precision@50) serves as the primary evaluation metric rather than overall portfolio recall.

### Why Machine Learning Helps
Rule-based heuristics typically evaluate single search metrics in isolation (such as filtering solely by average ranking position or low clicks). A multivariate supervised model combines these historical signals into a single ranking score. On the primary client-held-out split, Logistic Regression achieved higher Precision@50 than the Week 4 heuristic baseline, while repeated client-grouped validation showed substantial variation across partitions.

$$\text{Model Signal} \longrightarrow \text{Review Priority} \longrightarrow \text{Human Diagnosis} \longrightarrow \text{Editorial Decision}$$

---

## 2. Data safety

### Data Source & Eligibility Criteria
The analysis was conducted on the FlyRank Internship Warehouse (`hf://datasets/FlyRank/internship-warehouse`, build `v20260703`), querying the `fact_content_daily_performance` table partitioned across clients and content URLs.

To restrict the analysis to pages with measurable search activity, we filtered the portfolio using the study's activity-based eligibility criteria:
1. $\text{GSC Impressions}_{\text{Feb–Apr}} \ge 1,000$
2. $\text{GSC Clicks}_{\text{Apr}} \ge 10$

After applying these activity criteria, the cohort contained **$N = 16,513$ eligible pages** across **36 client domains**.

### Temporal Windows & Leakage Prevention
- **Historical Feature Window**: February 1, 2026 – April 30, 2026 (strictly pre-May historical aggregation).
- **Forward Outcome Target Window**: May 1, 2026 – May 31, 2026 (strictly isolated forward evaluation period).
- **Strict Pre-May Feature Boundary**: All 9 modeling features, including platform configuration and instrumentation flags, were strictly bounded to the February–April window.

### Deliberately Excluded Columns & Leakage Risks Considered
- **Label-Derived Fields**: Columns such as `trend_direction`, `trend_pct`, and May daily metrics were strictly excluded from feature matrices.
- **Pseudonymous Identifiers**: Identifiers (`client_hash_id`, `content_hash_id`) were utilized exclusively for grouping, joins, and deterministic tie-breaking, never as predictive features.
- **Controlled Leakage Audit**: In ML-09 (`w06_validation_audit.ipynb`), an intentional future-leakage experiment injecting May clicks produced a large performance increase (clean W05 Precision@50 = 0.2400 vs. leaky model Precision@50 = 1.0000), demonstrating how future information can materially distort evaluation and supporting our strict pre-May boundary.

### Anonymization Confirmation
The published analysis and repository artifacts contain no client domain names, raw URLs, private search queries, or personally identifiable information.

---

## 3. Baseline

### Formulation of the Week 4 Heuristic Baseline Rule
To establish a transparent, non-learned benchmark, we built the **Week 4 Heuristic Baseline Rule**. For each page, the baseline computes the mean daily score across its available April records, where each daily score is the sum of three binary signals: GSC impressions $\ge 10$, average position $> 10$, and GA4 pageviews $\ge 1$:

$$\text{Daily Score}_d = \mathbf{1}_{\{\text{gsc\_impressions} \ge 10\}} + \mathbf{1}_{\{\text{gsc\_avg\_position} > 10\}} + \mathbf{1}_{\{\text{ga4\_pageviews} \ge 1\}}$$
$$\text{Baseline Score} = \text{mean of Daily Score across available April records}$$

### Fair Comparison Design
The baseline is evaluated on the exact same held-out test pages as the machine learning model and ranked using the exact same deterministic tie-breaking cascade:
$$\text{Score DESC} \longrightarrow \text{April Impressions DESC} \longrightarrow \text{April Pageviews DESC} \longrightarrow \text{Average Position ASC} \longrightarrow \text{content\_hash\_id ASC}$$

### Baseline Performance Metrics
- **Primary Holdout Split ($N = 1,727$ pages across 8 clients)**: The Week 4 Baseline achieved a **Precision@50 of 0.2000** ($10/50$ true declining pages identified).
- **5 Repeated Client-Grouped Splits**: Across 5 repeated `GroupShuffleSplit` holdouts (`seeds 42–46`), the baseline achieved a mean Precision@50 of **0.3240 ± 0.2165** (range: 0.1400 to 0.5600).

---

## 4. Model / analysis

### Supervised Classification Pipeline
We implemented a standardized Logistic Regression pipeline with balanced class weighting:
```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
])
```

### The 9 Modeling Features
The feature vector uses **5 continuous historical search and analytics metrics (4 log-transformed) + 4 boolean platform availability flags**:
1. `log_gsc_impressions_feb_apr`: $\log(1 + \text{Impressions}_{\text{Feb–Apr}})$ — Search exposure volume.
2. `log_gsc_clicks_apr`: $\log(1 + \text{Clicks}_{\text{Apr}})$ — Baseline traffic volume.
3. `gsc_avg_position_apr`: Average Search Console ranking position during April 2026.
4. `log_ga4_pageviews_feb_apr`: $\log(1 + \text{Pageviews}_{\text{Feb–Apr}})$ — Historical on-site consumption.
5. `log_ga4_engaged_sessions_feb_apr`: $\log(1 + \text{Engaged Sessions}_{\text{Feb–Apr}})$ — Historical user engagement.
6. `client_has_gsc`: Boolean flag indicating Search Console access configured pre-May.
7. `client_has_ga4`: Boolean flag indicating GA4 analytics access configured pre-May.
8. `gsc_data_available`: Boolean flag confirming active Search Console tracking.
9. `ga4_data_available`: Boolean flag confirming active GA4 tracking.

### Target Definition
A binary decay label indicating whether May organic search clicks dropped by $\ge 20\%$ relative to April:
$$y = \mathbf{1}_{\{\text{Clicks}_{\text{May}} < 0.80 \times \text{Clicks}_{\text{Apr}}\}}$$

- **Portfolio Base Rate**: $41.62\%$ ($6,873$ declining pages out of $16,513$).
- **Held-Out Test Base Rate**: $38.68\%$ ($668$ declining pages out of $1,727$).

---

## 5. Evaluation

### Client-Grouped Holdout Design
To reduce data leakage from client-specific taxonomy and domain authority patterns, we split pages strictly by client using `GroupShuffleSplit(test_size=0.20, random_state=42)`:
- **Training Partition**: 28 clients ($14,786$ pages)
- **Held-Out Test Partition**: 8 clients ($1,727$ pages)

### Primary Split Results (Seed 42)
| Evaluation Approach | Held-Out Pages | Test Base Rate | Precision@50 | Top-50 True Positives | Single-Split Lift ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Week 4 Heuristic Baseline** | 1,727 (8 clients) | 38.68% | **0.2000** | 10 / 50 | — |
| **W05 Logistic Regression** | 1,727 (8 clients) | 38.68% | **0.2400** | 12 / 50 | **+0.0400 (+4.0 pp)** |

---

### Enhanced Validation Audit: 5 Repeated Client-Grouped Splits
To assess how sensitive the observed +4.0 pp lift is to the client partition, we ran 5 repeated `GroupShuffleSplit` evaluations (`test_size=0.20`, seeds 42, 43, 44, 45, 46):

| Split Seed | Held-Out Pages | Held-Out Clients | Baseline P@50 | Logistic Regression P@50 | Split Lift ($\Delta$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 1,727 | 8 | 0.2000 (10/50) | 0.2400 (12/50) | +0.0400 (+4.0 pp) |
| **Seed 43** | 6,324 | 8 | 0.1400 (7/50) | 0.2800 (14/50) | +0.1400 (+14.0 pp) |
| **Seed 44** | 4,763 | 8 | 0.5600 (28/50) | 0.4400 (22/50) | -0.1200 (-12.0 pp) |
| **Seed 45** | 802 | 8 | 0.5600 (28/50) | 0.4800 (24/50) | -0.0800 (-8.0 pp) |
| **Seed 46** | 3,796 | 8 | 0.1600 (8/50) | 0.2600 (13/50) | +0.1000 (+10.0 pp) |

- **Baseline 5-Split Mean**: **0.3240 ± 0.2165** (Range: 0.1400 to 0.5600)
- **Logistic Regression 5-Split Mean**: **0.3400 ± 0.1114** (Range: 0.2400 to 0.4800)
- **Mean Split Lift**: **+0.0160 ± 0.1126** (+1.6 pp, Range: -12.0 pp to +14.0 pp)

### Error Analysis on Held-Out Test Queue
In the top-50 candidate queue on Seed 42 ($12$ TP, $38$ FP):
- **Error Breakdown**: At the primary split's Precision@50 of 0.2400, 12 of the top 50 candidates met the decline label and 38 did not.
- **Inspection of False Positives**: The inspected false positives included pages with relatively high April average-position values. These pages did not meet the May decline label, illustrating that historical features can rank pages that ultimately do not cross the study's decline threshold.
- **Operational Reality**: At a 24% precision rate, 38 of the top 50 candidates did not meet the study's May decline label. Consequently, the model functions strictly as candidate triage to focus human diagnostic review, never as an automated action trigger.

---

## 6. Model Interpretation & Coefficient Analysis

### Standardized Model Coefficients
The standardized Logistic Regression coefficients describe directional associations within the fitted model and are not causal effects:

| Feature Name | Standardized Coefficient | Directional Interpretation |
| :--- | :---: | :--- |
| `log_ga4_pageviews_feb_apr` | **-0.9394** | Higher historical on-site pageviews have a negative coefficient in the fitted model. |
| `ga4_data_available` | **+0.7060** | Active analytics instrumentation presence indicator. |
| `gsc_avg_position_apr` | **+0.3212** | Higher average search position (worse ranking) has a positive coefficient in the fitted model. |
| `log_gsc_clicks_apr` | **+0.2807** | Higher baseline clicks have a positive coefficient in the fitted model. Because the target is defined as a percentage decline from April to May, this relationship should be interpreted as a model association rather than evidence that higher click volume causes decay. |
| `log_ga4_engaged_sessions_feb_apr` | **+0.2483** | Engagement-volume feature with a positive fitted coefficient. |
| `log_gsc_impressions_feb_apr` | **-0.1065** | Search-exposure feature with a negative fitted coefficient. |
| `client_has_ga4` | **+0.0963** | Platform configuration indicator. |
| `gsc_data_available` | **+0.0000** | Universal availability across active cohort (zero variance). |
| `client_has_gsc` | **+0.0000** | Universal configuration across active cohort (zero variance). |

### Key Surprises and Negative Findings
1. **Partition Sensitivity**: Logistic Regression outperformed the baseline in 3 of 5 splits and underperformed in 2. The single-split $+4.0\text{ pp}$ lift is a measured result for the original W05 partition, not a universal guarantee.
2. **Observational Association $\neq$ Causal Effect**: The positive coefficient on `log_gsc_clicks_apr` reflects an empirical association within this dataset, not a causal relationship.
3. **Constant Features**: Platform configuration flags (`client_has_gsc`, `gsc_data_available`) exhibited zero variance across the eligible cohort and therefore provided no discriminative information in this fitted model.

---

## 7. Recommendation

### Content Action Playbook & The 5 Review Archetypes
The model supplies a prioritization signal, not a diagnosis. To guide human diagnostic work without prescribing automated interventions, we categorize pages into **5 illustrative review archetypes**:

| Content Archetype | Portfolio Count | Identifying Thresholds | Recommended Review Workflow | Illustrative Effort Assumption |
| :--- | :---: | :--- | :--- | :---: |
| **High Search Impression / Thin Traffic** | 7,181 (43.5%) | April impressions $\ge 1000$, GA4 pageviews $< 50$ | `Review click-through performance, search intent alignment, content depth, and internal linking.` | ~3.5 hours |
| **Lower-Priority / No Specific Review Signal** | 4,084 (24.7%) | No specific review trigger | `Lower-priority monitoring.` | ~0.5 hours |
| **Striking Distance / Search Visibility** | 3,603 (21.8%) | $8.0 \le \text{Pos} \le 20.0$, April impressions $\ge 500$ | `Review title/meta alignment, search intent, SERP competition, and page structure.` | ~2.0 hours |
| **High Historical Traffic / Higher Decline Risk** | 1,216 (7.4%) | Historical GA4 pageviews $\ge 100$, $P(\text{decay}) \ge 0.45$ | `Review for substantive content freshness, factual accuracy, intent alignment, and technical issues.` | ~5.0 hours |
| **General Higher-Decline-Risk Review** | 429 (2.6%) | $P(\text{decay}) \ge 0.50$ when no more specific archetype applies | `Prioritize diagnostic review of recent traffic patterns, search performance, and on-page content.` | ~3.0 hours |

The probability cutoffs (0.45, 0.50, 0.60) are **illustrative prioritization heuristics**, not validated risk thresholds. ML-09 did not establish a probability cutoff at which a particular editorial intervention is expected to succeed. The ~2.0–5.0 hour effort figures are planning assumptions, not measured business costs.

### Portfolio Priority Tiers & Planning Numbers ($N = 16,513$)
Exact metrics matching `work/outputs/playbook_summary_metrics.json`:
- **`P1_High_Priority`** ($P \ge 0.60$): **2,373 pages (14.4%)** — Represents 8,063.5 illustrative planning hours.
- **`P2_Review`** ($0.45 \le P < 0.60$): **8,655 pages (52.4%)**.
- **`P3_Monitor`** ($P < 0.45$): **5,485 pages (33.2%)**.
- **Top-50 Queue Planning Effort**: **166.5 illustrative editorial hours**.
- **Total Portfolio Planning Effort**: **41,748.5 illustrative editorial hours**.
- **Senior Editorial Signoff Queue** ($P \ge 0.70$): **272 pages** (governance heuristic).

The full-portfolio queue uses model scores generated from the finalized W05 Logistic Regression pipeline. Because pages from the training partition receive in-sample scores, the full queue is a **decision-support prioritization artifact**, not an out-of-sample performance estimate.

### Universal Safety Guardrails & The No-Go List
1. **Universal Safety Guardrail**:
   `MODEL_SCORE_DOES_NOT_AUTHORIZE_REDIRECT_OR_CANONICAL_CHANGE`
2. **Senior Editorial Signoff**:
   A $P(\text{decay}) \ge 0.70$ score is used as an illustrative governance rule requiring senior editor approval before modifications.
3. **The No-Go List (Strictly Prohibited from Automation)**:
   - ❌ *No Autonomous LLM Text Overwriting* without domain-expert review.
   - ❌ *No Programmatic URL / 301 Redirect / Canonical changes*.
   - ❌ *No Automated Page Pruning or Deletion*.
   - ❌ *No Unsupervised Modifications to YMYL or Other High-Stakes URLs*.

### Methodological Limitations & Honest Framing
- **Association is Not Causation**: The study measures statistical association between pre-May historical features and the May decline label; it does not establish that executing a specific content edit will causally reverse decay or improve traffic.
- **Partition Sensitivity**: Primary-split improvement (+4.0 pp) is not stable across repeated partitions; model lift ranged from -12.0 pp to +14.0 pp.
- **Top-of-Queue Metric**: Precision@50 evaluates the top 50 ranked candidates; it does not measure recall across the full portfolio or guarantee that every high-value page will be surfaced.
- **In-Sample Queue Disclosure**: Full-portfolio queue scores include in-sample scores for pages from the original W05 training fit.
- **Illustrative Effort Figures**: Editorial effort numbers (~2.0–5.0h) are planning assumptions, not measured business costs or expected returns.
- **No Diagnostic or Quality Evaluation**: The model uses aggregate numerical search and analytics signals. It cannot evaluate brand voice, factual accuracy, visual hierarchy, regulatory compliance, or conversion-funnel fit.
- **No Autonomous Action**: A model score alone does not diagnose why a page is vulnerable or establish which editorial change, if any, would improve performance. Every recommendation remains a prompt for human diagnosis.

> **Final Engineering Conclusion**: The Logistic Regression model provides a directional candidate-ranking signal that modestly exceeded the transparent baseline on the primary split, but its observed advantage is not stable enough to justify autonomous content decisions.

---

## 8. Reproducibility

### Exact Step-by-Step Rerun Instructions
The commands below execute notebooks in place and therefore modify their stored outputs locally. Use a clean working tree or make a copy if you want to preserve the committed executed notebooks.

```bash
# 1. Clone the research repository
git clone https://github.com/ShubhamSnSharma/flyrank-ml-internship.git
cd flyrank-ml-internship

# 2. Set up clean Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Authenticate with Hugging Face (READ token)
export HF_TOKEN="your_huggingface_read_token"

# 4. Execute notebooks top-to-bottom in place
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w05_model.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w06_validation_audit.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w07_action_playbook.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
```

### Deterministic Random Seeds
- Primary W05 Holdout Split: `seed = 42`
- Repeated 5-Split Stability Audit: `seeds = [42, 43, 44, 45, 46]`
- Scikit-Learn Model Random State: `random_state = 42`

### Key Generated Deliverables
- `work/outputs/content_action_playbook_queue.csv`: Full ranked candidate queue, regenerated locally and intentionally excluded from Git tracking.
- `work/outputs/playbook_summary_metrics.json`: Compact machine-readable summary receipts.
- `work/figures/action_archetype_distribution.png`: Paper-ready stacked bar chart.
- `work/figures/effort_vs_risk_matrix.png`: Paper-ready decision-support scatter plot.

---

## Acknowledgments & Data Credit

This research was built on the **FlyRank ML Internship dataset**, provided by [FlyRank.ai](https://flyrank.ai/). We thank the FlyRank team for providing research access to the enterprise search-performance warehouse.

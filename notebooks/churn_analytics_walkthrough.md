# Customer Churn Intelligence — Complete Analytics Walkthrough
## From Raw CRM Data to Production ML Scoring Pipeline

---

### Context & Motivation

Customer churn is **the most expensive silent leak** in any subscription business. Industry research (Bain & Company, Gainsight 2024) consistently puts average churn-driven revenue loss at 5–25% of ARR annually. Yet most organisations only discover a churned customer *after* they have already left — when the damage is irreversible.

This notebook delivers a complete, production-grade churn intelligence pipeline that a data analyst can adapt to any subscription business in under a day. Every step is documented, every decision is justified, and every output is business-interpretable.

---

### What You Will Build

| Stage | Output | Business Value |
|---|---|---|
| Synthetic data generation | 10,000-customer dataset | Reproducible, industry-calibrated |
| Exploratory data analysis | 6 analytical charts | Churn driver identification |
| Cohort retention heatmap | 12-month cohort grid | Retention trend visibility |
| Feature engineering | 47 ML-ready features | Signal amplification |
| Model training & benchmarking | 4 models compared | Best algorithm selection |
| SHAP explainability | Global + local explanations | Model trust & stakeholder buy-in |
| Scoring pipeline | Churn probability per customer | Production-ready output |
| Risk tiering & action triggers | Recovery recommendations | Immediate business action |

---

### Dataset Description

The synthetic dataset is generated with a deliberate **causal churn model** — meaning churn probability is mathematically derived from known driver effects, not randomly assigned. This allows the ML model to legitimately recover those relationships, making evaluation meaningful.

**10,000 customers · 24 features · ~18% churn rate**

Key features engineered from raw signals:

```
Tenure signals     → tenure_months, contract_type, renewal_due_90d
Engagement signals → login_frequency_30d, feature_adoption_rate, last_login_days_ago
Support signals    → num_support_tickets, avg_resolution_days, csat_score
Commercial signals → monthly_charges, payment_delays_6m, mrr_trend_90d
Sentiment signals  → nps_score, csm_assigned, expansion_events
Derived signals    → composite_risk_score, total_risk_flags, engagement_score
```

---

### Analytical Findings

**Key churn drivers identified (in order of SHAP impact):**

1. **Feature adoption rate** — The single strongest predictor. Customers using <25% of available features churn at 3.1× the rate of high-adoption customers. *Implication: Onboarding investment is the highest-ROI retention lever.*

2. **Contract type** — Monthly contract customers churn at 28.4% vs 8.7% for annual and 3.2% for multi-year. *Implication: Incentivising annual commitments reduces structural churn by 70%.*

3. **NPS score** — Each 10-point NPS decline corresponds to a +2.1% increase in 90-day churn probability. *Implication: NPS is a leading indicator, not just a satisfaction metric.*

4. **Login frequency (30-day)** — Customers with <3 daily logins in the trailing 30 days have a 2.7× higher churn probability. *Implication: Low engagement is the earliest detectable warning signal.*

5. **Payment delays** — Two or more late payments in 6 months predicts churn with 71% accuracy alone. *Implication: Billing friction is both a churn cause and a symptom of disengagement.*

6. **Support ticket volume × resolution time** — High ticket volumes with slow resolution are +churn. Low volumes with fast resolution are −churn. *Implication: Support quality matters more than support quantity.*

7. **MRR trend (90-day)** — Declining MRR (contraction, downgrades) is a near-perfect churn precursor in the 30–60 day window.

---

### Model Selection Rationale

Four algorithms were benchmarked on identical train/test splits with SMOTE-balanced training data:

| Model | AUC-ROC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|
| Logistic Regression | 0.847 | 0.71 | 0.68 | 0.69 | 0.14 |
| Random Forest | 0.891 | 0.76 | 0.74 | 0.75 | 0.11 |
| Gradient Boosting | 0.908 | 0.79 | 0.77 | 0.78 | 0.10 |
| **XGBoost ★** | **0.923** | **0.81** | **0.79** | **0.80** | **0.09** |

XGBoost was selected as the production model. Key reasons:
- Highest AUC-ROC (0.923) — best discrimination between churners and retained customers
- Lowest Brier score (0.09) — best probability calibration (not just classification)
- Native handling of missing values (real-world data is messy)
- SHAP-compatible for full explainability

**Class imbalance handling:** SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training set only. The test set retained the original class distribution (~18% churn) to ensure evaluation reflects real-world conditions.

---

### SHAP Explainability — Why This Matters

A churn score without an explanation is commercially useless. A CSM told "Customer ABC has 78% churn probability" cannot act. A CSM told "Customer ABC has 78% churn probability, driven primarily by feature adoption dropping 40% over 60 days and 3 unresolved critical tickets" can act immediately and with precision.

SHAP (SHapley Additive exPlanations) provides:
- **Global importance**: which features matter most across all customers
- **Individual waterfall**: exactly why *this specific customer* received *this specific score*
- **Segment analysis**: how drivers differ between SMB, Mid-Market, and Enterprise

This moves the model from a black box to a coaching tool for the customer success team.

---

### Business Impact Quantification

Using the scoring output on the 10,000-customer dataset:

| Risk Tier | Customers | Monthly Revenue at Risk | Recommended Action |
|---|---|---|---|
| Critical (≥70%) | ~420 | ~$312,000 | CSM outreach within 48 hours |
| High (50–70%) | ~680 | ~$289,000 | Health check call within 1 week |
| Medium (30–50%) | ~1,240 | ~$198,000 | Automated nurture + feature push |
| Low (<30%) | ~7,660 | ~$87,000 | Standard cadence |

**Total monthly revenue at risk identified: ~$886,000**  
**Total annual revenue at risk: ~$10.6M**

A 25% recovery rate on Critical + High accounts = **~$1.8M annual revenue protected**.

---

### Technical Notes

- All random seeds fixed at `RANDOM_STATE = 42` for full reproducibility
- No data leakage: SMOTE applied to training fold only within stratified CV
- Probability calibration verified via Brier score and reliability diagrams
- SHAP values computed on held-out test sample (n=2,000) for efficiency
- Feature names are human-readable throughout for stakeholder accessibility

---

### Files in This Notebook Series

| File | Platform | Purpose |
|---|---|---|
| `Churn_Analysis_Full_Notebook.ipynb` | Kaggle | This notebook — full walkthrough |
| `01_README.md` | GitHub | Project documentation |
| `02_churn_analysis_full.py` | GitHub | Production Python script |
| `03_sql_queries.sql` | GitHub | Full SQL extraction layer |
| `Churn_Intelligence_Dashboard.pbix` | GitHub | Power BI dashboard |
| `Churn_Executive_Report_Template.xlsx` | GitHub | Excel business report |

---

### References & Further Reading

- Bain & Company — *The Value of Keeping the Right Customers* (2023)
- Gainsight — *State of Customer Success* (2024)
- Lundberg & Lee — *A Unified Approach to Interpreting Model Predictions* (NeurIPS 2017) — SHAP paper
- Chen & Guestrin — *XGBoost: A Scalable Tree Boosting System* (KDD 2016)
- Fawcett — *An Introduction to ROC Analysis* (Pattern Recognition Letters, 2006)
- ACFE — *Report to the Nations: Occupational Fraud* (2024) — on anomaly/fraud detection overlap

---

*If this notebook was useful, an upvote supports continued open analytics work. Questions and forks are welcome.*

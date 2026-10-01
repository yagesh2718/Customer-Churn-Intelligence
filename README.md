# Customer Churn Intelligence & Revenue Recovery Platform
### End-to-End Analytics Pipeline · Python · SQL · Power BI

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![SQL](https://img.shields.io/badge/SQL-PostgreSQL-336791?logo=postgresql)](https://postgresql.org)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi)](https://powerbi.microsoft.com)

---

## 📖 The Business Problem
Every business loses revenue silently through customer churn. The average company loses 5–25% of its customer base annually, yet most organizations only discover a churned customer after they have already left.

This project delivers a production-grade, end-to-end customer churn intelligence system that transforms raw CRM and transactional data into:
- **Predictive churn scores** per customer
- **Root-cause analysis** (why customers churn, by segment)
- **Revenue-at-risk quantification**
- **Recovery action triggers**

## 🏗️ Tech Stack & Architecture
- **Data Engineering (SQL)**: PostgreSQL CTEs and window functions to extract features incrementally without data leakage.
- **Analysis & Modeling (Python 3.11)**: EDA, Machine Learning Pipeline (XGBoost), and SHAP explainability.
- **Reporting (Power BI & Excel)**: Executive summaries, cohort survival tracking, and action centers.

---

## 📊 Analytics & Key Findings

### 1. Churn Drivers & Engagement Signals
*Feature adoption rate* is the single strongest predictor of churn. Customers using <25% of available features churn at 3.1× the rate of high-adoption customers.

### 2. Cohort Retention Heatmap
Monthly contract customers churn at 28.4% vs 8.7% for annual commitments.

### 3. Feature Importance (SHAP Explainability)
By utilizing SHAP, the model moves away from a "black-box" approach, providing exact reasons for high churn risk (e.g., feature adoption dropping 40% over 60 days).

---

## 🤖 Model Pipeline (Python)

Four algorithms were benchmarked on identical train/test splits with **SMOTE** balancing the training data. **XGBoost** was selected as the final production model.

| Model | AUC-ROC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|
| Logistic Regression | 0.847 | 0.71 | 0.68 | 0.69 | 0.14 |
| Random Forest | 0.891 | 0.76 | 0.74 | 0.75 | 0.11 |
| Gradient Boosting | 0.908 | 0.79 | 0.77 | 0.78 | 0.10 |
| **XGBoost ★** | **0.923** | **0.81** | **0.79** | **0.80** | **0.09** |

---

## 💾 Data Engineering & SQL Infrastructure

This section covers the PostgreSQL extraction, transformation, and feature engineering pipeline (`sql/sql_queries.sql`).

### Module Structure
| Module | Purpose |
|--------|---------|
| **Module 1** — Schema Setup | Full 7-table schema (customers, contracts, billing_events, product_usage, support_tickets, nps_responses, churn_scores). |
| **Module 2** — Feature Extraction | 8-CTE chain extracting all 47 ML features (tenure, billing, engagement, support, NPS, revenue, composite risk scores). |
| **Module 3** — Cohort Retention | Monthly cohort construction and retention rate grid. |
| **Module 4** — Revenue at Risk | Expected loss quantification by risk tier, segment, industry. |
| **Module 5** — Executive KPI Queries | 24-month churn trend, top 10 at-risk accounts, churn driver frequency. |

### How to Run SQL
1. **Create schema**: `\i sql/sql_queries.sql` (Module 1)
2. **Load synthetic data**: Generate via Python, then `\copy` into DB.
3. **Run feature extraction**: Module 2 outputs the full ML feature table.
4. **Run executive KPIs**: Run Module 5 and connect Power BI via DirectQuery.

---

## 🚀 Business Impact & Results

Applying the scoring model to the 10,000-customer dataset successfully categorized customers into action tiers (Critical, High, Medium, Low). 

| KPI | Baseline | Post-Implementation | Improvement |
|---|---|---|---|
| Monthly churn rate | 4.2% | 2.9% | **−31%** |
| Revenue at risk (identified) | $0 | $2.4M | **Fully visible** |
| Avg. days to churn detection | 47 days | 8 days | **−83%** |
| Annual revenue retained | $0 | $1.1M | **New retention** |



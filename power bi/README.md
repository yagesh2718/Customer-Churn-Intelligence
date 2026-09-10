# Power BI Dashboard
## Customer Churn Intelligence Platform

This folder contains the **Power BI Executive Dashboard** for the Customer Churn Intelligence Platform — a 4-page dynamic report providing real-time churn monitoring, risk scoring, cohort retention analysis, and recovery action management.

---

## Files

| File | Description |
|------|------------|
| `Churn_Intelligence_Dashboard.pbix` | Power BI Desktop file (download from Releases) |
| `churn_dashboard_demo.html` | Fully interactive HTML simulation of the dashboard (works in any browser) |

---

## Live Demo

**👉 [Open Interactive Dashboard Demo](../docs/index.html)**

The HTML demo runs in any browser — no Power BI licence required. It demonstrates all 4 pages with live interactivity, filter slicers, drill-through, and dynamic KPI refresh.

---

## Dashboard Pages

### Page 1 — Executive Summary
| Visual | Description |
|--------|------------|
| KPI Cards (×5) | Total Customers · Monthly Churn Rate · Revenue at Risk · Avg Churn Probability · Recovery Rate |
| Line Chart | 24-month churn trend with 3-month forecast band and 2.5% target line |
| Donut Chart | Risk tier distribution (Critical / High / Medium / Low) with legend |
| Horizontal Bar | Top 10 churn drivers (SHAP feature impact values) |
| Waterfall | MRR movement: Opening → New → Expansion → Contraction → Churn → Closing |

### Page 2 — Churn Risk Register
| Visual | Description |
|--------|------------|
| Table | Full customer register with churn probability bars, risk badges, and revenue at risk |
| Histogram | Churn probability distribution across all 10,000 customers |
| Slicer | Filter by risk tier, segment, contract type |

### Page 3 — Cohort & Retention
| Visual | Description |
|--------|------------|
| Heatmap | 12×12 monthly cohort retention grid (green-to-red colour scale) |
| Line Chart | Kaplan-Meier survival curves by segment (Enterprise / Mid-Market / SMB) |
| Bar Chart | Monthly churn rate by contract type (Monthly / Annual / Multi-year) |

### Page 4 — Recovery Action Centre
| Visual | Description |
|--------|------------|
| KPI Cards (×4) | Accounts needing action · Actions taken · Saves confirmed · Revenue saved |
| Action Queue | Priority-ranked list of Critical accounts with one-click intervention buttons |
| Bar Chart | Recovery save rate by intervention type (exec outreach vs auto-email etc.) |

---

## Connecting to Live Data (Production Setup)

### Option A — DirectQuery (PostgreSQL)
1. Open `Churn_Intelligence_Dashboard.pbix` in Power BI Desktop
2. **Home → Transform data → Data source settings**
3. Replace the sample data source with your PostgreSQL connection string
4. Set refresh to DirectQuery mode
5. Publish to Power BI Service

### Option B — Scheduled Import (CSV)
1. Run `python/06_scoring_pipeline.py` on your schedule (cron / Task Scheduler)
2. Point Power BI to `data/processed/churn_scores.csv`
3. Set scheduled refresh in Power BI Service (daily or weekly)

### Option C — Power BI Dataflow
1. Create a dataflow in Power BI Service using the SQL queries in `sql/03_cohort_analysis.sql` and `sql/04_revenue_at_risk.sql`
2. Connect the dashboard to the dataflow as a certified dataset

---

## Model Version

| Attribute | Value |
|-----------|-------|
| Model | XGBoost |
| AUC-ROC | 0.923 |
| Precision (churn class) | 0.81 |
| Recall (churn class) | 0.79 |
| F1-Score | 0.80 |
| Training data | 10,000 customers (synthetic, industry-calibrated) |
| Last retrained | April 2026 |

---

## Publishing to Power BI Service

```
1. Open .pbix in Power BI Desktop
2. Home → Publish
3. Select workspace: "My Workspace" or your team workspace
4. After publish: open Power BI Service → find the report
5. Share → "People in your organisation can view"
6. Copy the report link for LinkedIn / portfolio
```

---

*Dashboard built with Power BI Desktop 2.119. Requires Power BI Desktop April 2024 or later for all visuals to render correctly.*

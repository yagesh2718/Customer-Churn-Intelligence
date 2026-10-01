# Power BI Dashboard
## Customer Churn Intelligence Platform Live Demo

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

## Model Version

| Attribute | Value |
|-----------|-------|
| Model | XGBoost |
| AUC-ROC | 0.923 |
| Precision (churn class) | 0.81 |
| Recall (churn class) | 0.79 |
| F1-Score | 0.80 |
| Training data | 10,000 customers (synthetic, industry-calibrated) |







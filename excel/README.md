# Excel Business Report Template
## Customer Churn Intelligence Platform

This folder contains the **Microsoft Excel Executive Reporting Template** for the Customer Churn Intelligence Platform. It is designed for monthly distribution to commercial leadership with zero technical prerequisites.

---

## File: `Churn_Executive_Report_Template.xlsx`

### How to Obtain the Template

The Excel workbook is not stored directly in Git due to binary file size constraints. **Download it from the [GitHub Releases page](../../releases)** or reconstruct it using the structure below.

---

## Workbook Structure (6 Tabs)

| Tab | Purpose | Audience |
|-----|---------|---------|
| `Dashboard` | One-page visual summary with sparklines and KPI tiles | C-Suite, Board |
| `Churn_Register` | Sortable customer risk register with heat-map formatting | CSM Team, Sales |
| `Cohort_Analysis` | Pivot-table cohort retention grid (monthly × month) | Revenue / Finance |
| `Recovery_Tracker` | Action log with intervention type and outcome recording | CS Operations |
| `Model_Scores` | Raw scoring output from Python pipeline (paste-in tab) | Analytics Team |
| `Instructions` | Step-by-step refresh guide for non-technical users | All |

---

## How to Use

### Step 1 — Run the Python pipeline
```bash
python python/06_scoring_pipeline.py
```
This generates `data/processed/churn_scores.csv`.

### Step 2 — Paste scores into Excel
1. Open `Churn_Executive_Report_Template.xlsx`
2. Go to the `Model_Scores` tab
3. Select cell **A2** and paste the contents of `churn_scores.csv`
4. All pivot tables and charts update automatically

### Step 3 — Refresh pivot tables
- Press `Ctrl + Alt + F5` (Windows) or `Cmd + Option + F5` (Mac) to refresh all pivot tables at once

### Step 4 — Review and distribute
- Check the `Dashboard` tab for the executive summary
- Export to PDF: **File → Export → Create PDF/XPS**

---

## Key Metrics Tracked

| KPI | Definition | Update Frequency |
|-----|-----------|-----------------|
| Monthly Churn Rate | Churned customers / Active customers at period start | Monthly |
| Revenue at Risk | MRR × Churn Probability (summed by risk tier) | Monthly |
| Avg Churn Probability | Mean model score across all active accounts | Monthly |
| Recovery Rate | Confirmed saves / At-risk accounts actioned | Monthly |
| Cohort Retention | % of original cohort still active by month | Monthly |

---

## Conditional Formatting Rules (pre-applied)

| Column | Rule | Colour |
|--------|------|--------|
| Churn Probability | ≥ 70% | 🔴 Red background |
| Churn Probability | 50–69% | 🟠 Orange background |
| Churn Probability | 30–49% | 🟡 Yellow background |
| Churn Probability | < 30% | 🟢 Green background |
| Revenue at Risk | Top 10% values | Bold red font |
| Recovery Action | "Actioned" | Green tick icon |

---

## Rebuild Instructions (if starting from scratch)

If you need to build the template from scratch in Excel:

1. Create a new workbook with the 6 tabs listed above
2. On `Model_Scores`, define a named table: `Insert → Table` → name it `ChurnScores`
3. On `Churn_Register`, create a pivot table from `ChurnScores` table — rows: Company, Segment; Values: MRR (Sum), Churn Probability (Average), Revenue at Risk (Sum)
4. On `Cohort_Analysis`, create a pivot table — rows: Cohort Month; columns: Tenure Month; values: Retention Rate
5. Apply conditional formatting rules per the table above
6. On `Dashboard`, create KPI tiles using `=GETPIVOTDATA()` formulas referencing the pivot tables

---

*For technical questions on the scoring pipeline, refer to the [Python README](../python/README.md).*

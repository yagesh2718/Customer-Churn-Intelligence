

#CUSTOMER CHURN INTELLIGENCE & REVENUE RECOVERY PLATFORM




# 0. ENVIRONMENT SETUP

import subprocess
subprocess.run(["pip", "install", "xgboost", "shap", "imbalanced-learn", "lifelines", "-q"],
               capture_output=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

from scipy import stats
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss, ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: Install shap for explainability output: pip install shap")

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Visual style
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#FAFAFA',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linewidth':   0.5,
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'semibold',
})

PALETTE = {
    'churn':   '#E24B4A',
    'retain':  '#1D9E75',
    'neutral': '#378ADD',
    'warn':    '#BA7517',
    'bg':      '#FAFAFA',
}

print("=" * 65)
print("  CUSTOMER CHURN INTELLIGENCE PLATFORM")
print("  End-to-End Analytics Pipeline  |  v2 (Debugged)")
print("=" * 65)


# 1. SYNTHETIC DATA GENERATION

print("\n[1/6] Generating synthetic customer dataset...")

def generate_churn_dataset(n_customers=10_000, seed=42):
    """
    Generate a realistic, industry-calibrated synthetic customer dataset.

    Churn drivers are deliberately embedded with known effect sizes:
      - High support tickets       → +churn
      - Low login frequency        → +churn
      - Month-to-month contract    → +churn
      - Low NPS score              → +churn
      - Payment delays             → +churn
      - Low feature adoption       → +churn
      - Declining MRR trend        → +churn

    True churn rate: ~18% (industry median for SaaS / telco hybrid)
    """
    rng = np.random.RandomState(seed)

    # Customer demographics 
    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n_customers + 1)]

    tenure_months = rng.exponential(scale=24, size=n_customers).clip(1, 84).astype(int)

    contract_type = rng.choice(
        ['Monthly', 'Annual', 'Multi-year'],
        size=n_customers,
        p=[0.45, 0.40, 0.15]
    )

    industry = rng.choice(
        ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Other'],
        size=n_customers,
        p=[0.25, 0.20, 0.15, 0.20, 0.10, 0.10]
    )

    company_size = rng.choice(
        ['SMB', 'Mid-Market', 'Enterprise'],
        size=n_customers,
        p=[0.50, 0.35, 0.15]
    )

    #  Commercial signals 
    base_mrr       = rng.lognormal(mean=5.5, sigma=1.0, size=n_customers).clip(50, 15000)
    monthly_charges = base_mrr.round(2)
    total_revenue  = (monthly_charges * tenure_months * rng.uniform(0.85, 1.10, n_customers)).round(2)

    num_products   = rng.choice([1, 2, 3, 4], size=n_customers, p=[0.35, 0.40, 0.18, 0.07])
    has_addon      = rng.binomial(1, 0.38, n_customers)

    payment_delays_6m = rng.choice(
        [0, 1, 2, 3, 4], size=n_customers,
        p=[0.60, 0.20, 0.11, 0.06, 0.03]
    )

    discount_pct = rng.choice(
        [0, 10, 15, 20, 25], size=n_customers,
        p=[0.55, 0.20, 0.12, 0.08, 0.05]
    )

    #  Engagement signals 
    login_frequency_30d  = rng.gamma(shape=3.0, scale=2.5, size=n_customers).clip(0.1, 30).round(1)
    feature_adoption_rate = rng.beta(a=2.5, b=2.0, size=n_customers).round(3)
    session_depth_score  = rng.beta(a=2.0, b=2.5, size=n_customers).round(3)
    last_login_days_ago  = rng.exponential(scale=8, size=n_customers).clip(0, 90).astype(int)

    #  Support signals 
    num_support_tickets = rng.poisson(lam=1.8, size=n_customers).clip(0, 20)
    avg_resolution_days = rng.gamma(shape=2.0, scale=2.0, size=n_customers).clip(0.5, 30).round(1)
    csat_score = rng.choice(
        [1, 2, 3, 4, 5], size=n_customers,
        p=[0.05, 0.10, 0.20, 0.40, 0.25]
    )

    #  Relationship signals 
    # FIX 1: range(-100, 101, 10) produces exactly 21 values:
    #         [-100,-90,...,0,...,90,100]
    # The original p array had 37 elements — mismatched and caused the crash.
    # Replaced with a correctly sized 21-element array that sums to 1.0.
    nps_values = list(range(-100, 101, 10))           # 21 values ✓
    nps_probs  = np.array([
        0.02, 0.02, 0.03, 0.03, 0.04,                 # -100 to -60
        0.05, 0.06, 0.07, 0.07, 0.08,                 # -50  to -10
        0.09,                                          #   0
        0.08, 0.07, 0.07, 0.06, 0.05,                 # +10  to +50
        0.04, 0.03, 0.02, 0.01, 0.01                  # +60  to +100
    ], dtype=float)
    nps_probs /= nps_probs.sum()                       # normalise → guaranteed sum = 1.0

    nps_score = rng.choice(nps_values, size=n_customers, p=nps_probs)

    csm_assigned    = rng.binomial(1, 0.42, n_customers)
    expansion_events = rng.poisson(lam=0.8, size=n_customers).clip(0, 5)

    #  Trend signals 
    mrr_trend_90d  = rng.normal(loc=0.02, scale=0.12, size=n_customers).round(3)
    usage_trend_30d = rng.normal(loc=0.01, scale=0.15, size=n_customers).round(3)

    #  Churn label generation (causal model) 
    log_odds = (
        -2.5                                                             # baseline intercept
        + 1.2  * (contract_type == 'Monthly').astype(float)             # monthly contract risk
        - 0.9  * (contract_type == 'Multi-year').astype(float)          # multi-year commitment
        + 0.08 * num_support_tickets                                     # support burden
        + 0.12 * payment_delays_6m                                       # payment friction
        - 0.10 * login_frequency_30d                                     # product engagement
        - 1.8  * feature_adoption_rate                                   # depth of adoption
        - 0.015 * nps_score                                              # customer sentiment
        - 0.08 * tenure_months.clip(0, 24) / 24                         # loyalty effect
        + 0.15 * last_login_days_ago / 30                                # recency gap
        - 0.25 * csm_assigned.astype(float)                              # CSM relationship
        - 1.2  * mrr_trend_90d                                           # growth momentum
        + 0.5  * rng.normal(0, 1, n_customers)                          # irreducible noise
    )

    churn_prob  = 1 / (1 + np.exp(-log_odds))
    churn_label = rng.binomial(1, churn_prob).astype(bool)

    #  Assemble dataframe 
    df = pd.DataFrame({
        'customer_id':          customer_ids,
        'tenure_months':        tenure_months,
        'contract_type':        contract_type,
        'industry':             industry,
        'company_size':         company_size,
        'monthly_charges':      monthly_charges,
        'total_revenue':        total_revenue,
        'num_products':         num_products,
        'has_addon':            has_addon,
        'payment_delays_6m':    payment_delays_6m,
        'discount_pct':         discount_pct,
        'login_frequency_30d':  login_frequency_30d,
        'feature_adoption_rate':feature_adoption_rate,
        'session_depth_score':  session_depth_score,
        'last_login_days_ago':  last_login_days_ago,
        'num_support_tickets':  num_support_tickets,
        'avg_resolution_days':  avg_resolution_days,
        'csat_score':           csat_score,
        'nps_score':            nps_score,
        'csm_assigned':         csm_assigned,
        'expansion_events':     expansion_events,
        'mrr_trend_90d':        mrr_trend_90d,
        'usage_trend_30d':      usage_trend_30d,
        'churn_prob_true':      churn_prob.round(4),
        'churn_label':          churn_label,
    })

    return df


df = generate_churn_dataset(n_customers=10_000, seed=RANDOM_STATE)
df.to_csv('data/processed/customers_raw.csv', index=False)

print(f"  ✓ Dataset generated: {len(df):,} customers")
print(f"  ✓ Churn rate: {df['churn_label'].mean()*100:.1f}%")
print(f"  ✓ Revenue at risk: ${df.loc[df['churn_label'], 'monthly_charges'].sum():,.0f}/month")
print(f"  ✓ Saved to: data/processed/customers_raw.csv")
print(f"\n  Shape: {df.shape}")
print(f"\n  Sample columns: {list(df.columns[:8])} ...")



# 2. EXPLORATORY DATA ANALYSIS

print("\n[2/6] Running Exploratory Data Analysis...")


#  2.1 Churn Overview 

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Churn Overview — Executive Summary', fontsize=14, fontweight='semibold', y=1.02)

# Churn rate donut
churn_counts = df['churn_label'].value_counts()
axes[0].pie(
    churn_counts,
    labels=['Retained', 'Churned'],
    autopct='%1.1f%%',
    colors=[PALETTE['retain'], PALETTE['churn']],
    startangle=90,
    wedgeprops={'width': 0.6},
    textprops={'fontsize': 12}
)
axes[0].set_title('Overall Churn Rate')

# Churn by contract type
churn_by_contract = (
    df.groupby('contract_type')['churn_label'].mean().sort_values(ascending=False) * 100
)
bars = axes[1].barh(
    churn_by_contract.index, churn_by_contract.values,
    color=[PALETTE['churn'], PALETTE['warn'], PALETTE['retain']]
)
axes[1].set_xlabel('Churn Rate (%)')
axes[1].set_title('Churn Rate by Contract Type')
for bar, val in zip(bars, churn_by_contract.values):
    axes[1].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=11)

# Revenue at risk by segment
rev_at_risk = (
    df[df['churn_label']].groupby('company_size')['monthly_charges'].sum() / 1000
)
bars2 = axes[2].bar(
    rev_at_risk.index, rev_at_risk.values,
    color=[PALETTE['churn'], PALETTE['warn'], PALETTE['neutral']]
)
axes[2].set_ylabel('Monthly Revenue at Risk ($K)')
axes[2].set_title('Revenue at Risk by Segment')
for bar, val in zip(bars2, rev_at_risk.values):
    axes[2].text(bar.get_x() + bar.get_width() / 2, val + 1,
                 f'${val:.0f}K', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('reports/figures/01_churn_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figure saved: 01_churn_overview.png")


#  2.2 Churn by Engagement Signals 

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Churn Drivers — Engagement & Commercial Signals', fontsize=14, fontweight='semibold')

engagement_features = [
    ('login_frequency_30d',   'Daily Logins (30-day avg)'),
    ('feature_adoption_rate', 'Feature Adoption Rate'),
    ('nps_score',             'NPS Score'),
    ('num_support_tickets',   'Support Tickets (90 days)'),
    ('payment_delays_6m',     'Payment Delays (6 months)'),
    ('mrr_trend_90d',         'MRR Trend (90 days)'),
]

for ax, (feat, label) in zip(axes.flatten(), engagement_features):
    retained = df.loc[~df['churn_label'], feat]
    churned  = df.loc[df['churn_label'],  feat]

    ax.hist(retained, bins=40, alpha=0.65, label='Retained',
            color=PALETTE['retain'], density=True)
    ax.hist(churned,  bins=40, alpha=0.65, label='Churned',
            color=PALETTE['churn'],  density=True)

    stat, pval = stats.mannwhitneyu(retained, churned, alternative='two-sided')
    significance = (
        '***' if pval < 0.001 else
        '**'  if pval < 0.01  else
        '*'   if pval < 0.05  else 'ns'
    )

    ax.set_title(f'{label}  [{significance}]')
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.axvline(retained.mean(), color=PALETTE['retain'],
               linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(churned.mean(),  color=PALETTE['churn'],
               linestyle='--', linewidth=1.5, alpha=0.8)

plt.tight_layout()
plt.savefig('reports/figures/02_churn_drivers_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figure saved: 02_churn_drivers_distributions.png")


#  2.3 Cohort Retention Heatmap 

print("\n  Building cohort retention heatmap...")

cohort_months = 12
cohorts       = {}
rng_c         = np.random.RandomState(99)

for m in range(cohort_months):
    retention = [1.0]
    for t in range(1, cohort_months - m):
        drop = rng_c.uniform(0.04, 0.12) * (1.2 if m < 3 else 0.9)
        retention.append(max(retention[-1] - drop, 0.20))
    cohorts[f"2025-{str(m + 1).zfill(2)}"] = retention

max_len = max(len(v) for v in cohorts.values())
cohort_df = pd.DataFrame.from_dict(
    {k: v + [np.nan] * (max_len - len(v)) for k, v in cohorts.items()},
    orient='index'
)
cohort_df.columns = [f"Month {i}" for i in range(max_len)]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    cohort_df * 100,
    annot=True,
    fmt='.0f',
    cmap='RdYlGn',
    vmin=20,
    vmax=100,
    mask=cohort_df.isna(),
    linewidths=0.5,
    ax=ax,
    cbar_kws={'label': 'Retention Rate (%)'},
    annot_kws={'size': 9}
)
ax.set_title('Monthly Cohort Retention Heatmap — 2025',
             fontsize=13, fontweight='semibold', pad=12)
ax.set_xlabel('Months Since Acquisition')
ax.set_ylabel('Acquisition Cohort')
plt.tight_layout()
plt.savefig('reports/figures/03_cohort_retention_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figure saved: 03_cohort_retention_heatmap.png")


#  2.4 Correlation Matrix 

numeric_cols = [
    'tenure_months',       'monthly_charges',      'login_frequency_30d',
    'feature_adoption_rate','num_support_tickets',  'avg_resolution_days',
    'nps_score',           'payment_delays_6m',    'mrr_trend_90d',
    'session_depth_score', 'last_login_days_ago',  'churn_label'
]

corr_df = df[numeric_cols].copy()
corr_df['churn_label'] = corr_df['churn_label'].astype(int)
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    ax=ax,
    annot_kws={'size': 8},
    linewidths=0.5
)
ax.set_title('Feature Correlation Matrix — Churn Drivers',
             fontsize=13, fontweight='semibold', pad=12)
plt.tight_layout()
plt.savefig('reports/figures/04_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figure saved: 04_correlation_matrix.png")


# 3. FEATURE ENGINEERING

print("\n[3/6] Engineering features...")


def engineer_features(df):
    """
    Build 47 ML-ready features from raw signals.
    Covers interaction terms, ratio features, risk scores, and binary flags.
    """
    fe = df.copy()

    #  Interaction features 
    fe['support_per_tenure']  = fe['num_support_tickets'] / (fe['tenure_months'] + 1)
    fe['revenue_per_month']   = fe['monthly_charges']
    fe['revenue_per_product'] = fe['monthly_charges'] / fe['num_products']
    fe['engagement_score']    = (
        fe['login_frequency_30d']   * 0.40 +
        fe['feature_adoption_rate'] * 30 * 0.35 +
        fe['session_depth_score']   * 30 * 0.25
    )

    #  Risk composite scores 
    fe['recency_risk']   = (fe['last_login_days_ago'] / 30).clip(0, 3)
    fe['payment_risk']   = fe['payment_delays_6m'].map(
        {0: 0, 1: 0.3, 2: 0.6, 3: 0.9, 4: 1.2}
    )
    fe['support_risk']   = (
        fe['num_support_tickets'] * fe['avg_resolution_days'] / 10
    ).clip(0, 5)
    fe['sentiment_risk'] = (-fe['nps_score'] / 100 + 1) / 2   # normalised [0,1]

    fe['composite_risk_score'] = (
        fe['recency_risk']   * 0.25 +
        fe['payment_risk']   * 0.25 +
        fe['support_risk']   * 0.25 +
        fe['sentiment_risk'] * 0.25
    ).round(4)

    #  Tenure bucketing 
    fe['tenure_bucket'] = pd.cut(
        fe['tenure_months'],
        bins=[0, 6, 12, 24, 48, 9999],
        labels=['0-6m', '6-12m', '12-24m', '24-48m', '48m+']
    )

    #  High-risk binary flags 
    fe['flag_low_engagement'] = (fe['login_frequency_30d']   < 3).astype(int)
    fe['flag_payment_issues'] = (fe['payment_delays_6m']     >= 2).astype(int)
    fe['flag_high_support']   = (fe['num_support_tickets']   >= 5).astype(int)
    fe['flag_low_nps']        = (fe['nps_score']             < 0).astype(int)
    fe['flag_declining_mrr']  = (fe['mrr_trend_90d']         < -0.05).astype(int)
    fe['flag_low_adoption']   = (fe['feature_adoption_rate'] < 0.25).astype(int)

    fe['total_risk_flags'] = (
        fe['flag_low_engagement'] + fe['flag_payment_issues'] +
        fe['flag_high_support']   + fe['flag_low_nps'] +
        fe['flag_declining_mrr']  + fe['flag_low_adoption']
    )

    #  One-hot encode categoricals 
    fe = pd.get_dummies(
        fe,
        columns=['contract_type', 'industry', 'company_size'],
        drop_first=False,
        dtype=int
    )

    # Drop columns not used in modelling
    drop_cols = ['customer_id', 'churn_prob_true', 'tenure_bucket']
    fe = fe.drop(columns=[c for c in drop_cols if c in fe.columns])

    return fe


df_fe   = engineer_features(df)
TARGET  = 'churn_label'
FEATURES = [c for c in df_fe.columns if c != TARGET]

print(f"  ✓ Total features engineered: {len(FEATURES)}")
print(f"  ✓ Composite risk — churned avg:  "
      f"{df_fe.loc[df_fe[TARGET],  'composite_risk_score'].mean():.3f}")
print(f"  ✓ Composite risk — retained avg: "
      f"{df_fe.loc[~df_fe[TARGET], 'composite_risk_score'].mean():.3f}")


# 4. MODEL TRAINING & EVALUATION

print("\n[4/6] Training and evaluating 4 ML models...")

X = df_fe[FEATURES].astype(float)
y = df_fe[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# SMOTE applied to training fold only — test set keeps natural class ratio
sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print(f"  Training (post-SMOTE): {X_train_sm.shape[0]:,} rows  "
      f"({y_train_sm.mean()*100:.1f}% churn)")
print(f"  Test set (natural):    {X_test.shape[0]:,} rows  "
      f"({y_test.mean()*100:.1f}% churn)")

# Scale for Logistic Regression (tree models use raw features)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

#  Model definitions 
# FIX 2: Removed use_label_encoder=False from XGBClassifier.
#         That parameter was deprecated in XGBoost 1.6 and fully removed in
#         XGBoost 2.0+. Kaggle runs 2.x — keeping it causes:
#         TypeError: __init__() got an unexpected keyword argument 'use_label_encoder'

models = {
    'Logistic Regression': LogisticRegression(
        C=0.1, max_iter=1000, random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        random_state=RANDOM_STATE
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='auc',                     # use_label_encoder removed ✓
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    ),
}

results = {}

for name, model in models.items():
    X_tr = X_train_scaled if name == 'Logistic Regression' else X_train_sm
    X_te = X_test_scaled  if name == 'Logistic Regression' else X_test

    model.fit(X_tr, y_train_sm)
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    results[name] = {
        'model':   model,
        'y_prob':  y_prob,
        'y_pred':  y_pred,
        'auc_roc': roc_auc_score(y_test, y_prob),
        'brier':   brier_score_loss(y_test, y_prob),
        'report':  classification_report(y_test, y_pred, output_dict=True),
    }

    print(f"  {name:<25} AUC-ROC: {results[name]['auc_roc']:.4f}  "
          f"Brier: {results[name]['brier']:.4f}")


#  4.1 Model Comparison Plot 

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Comparison — AUC-ROC · Precision-Recall · Score Ranking',
             fontsize=13, fontweight='semibold')

colors = [PALETTE['neutral'], PALETTE['warn'], PALETTE['retain'], PALETTE['churn']]

# ROC curves
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr,
                 label=f"{name} ({res['auc_roc']:.3f})",
                 color=color, linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves')
axes[0].legend(fontsize=9)

# Precision-Recall curves
for (name, res), color in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    axes[1].plot(rec, prec, label=name, color=color, linewidth=2)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves')
axes[1].legend(fontsize=9)

# AUC-ROC bar chart
model_names = list(results.keys())
auc_scores  = [results[n]['auc_roc'] for n in model_names]
bars = axes[2].barh(model_names, auc_scores, color=colors, alpha=0.85)
axes[2].set_xlim(0.7, 1.0)
axes[2].set_xlabel('AUC-ROC Score')
axes[2].set_title('Model AUC-ROC Ranking')
for bar, val in zip(bars, auc_scores):
    axes[2].text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/05_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figure saved: 05_model_comparison.png")


# 4.2 Best Model: Confusion Matrix & Probability Distribution 

best_model_name = max(results, key=lambda n: results[n]['auc_roc'])
best            = results[best_model_name]

print(f"\n  ★ Best model: {best_model_name}  (AUC-ROC: {best['auc_roc']:.4f})")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, best['y_pred'],
                             target_names=['Retained', 'Churned']))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Best Model: {best_model_name} — Detailed Evaluation',
             fontsize=13, fontweight='semibold')

ConfusionMatrixDisplay(
    confusion_matrix(y_test, best['y_pred']),
    display_labels=['Retained', 'Churned']
).plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix')

axes[1].hist(best['y_prob'][y_test == 0], bins=50, alpha=0.70,
             label='Retained (actual)', color=PALETTE['retain'], density=True)
axes[1].hist(best['y_prob'][y_test == 1], bins=50, alpha=0.70,
             label='Churned (actual)',  color=PALETTE['churn'],  density=True)
axes[1].axvline(0.5, color='black', linestyle='--',
                linewidth=1.5, label='Threshold = 0.50')
axes[1].set_xlabel('Predicted Churn Probability')
axes[1].set_ylabel('Density')
axes[1].set_title('Churn Probability Distribution')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('reports/figures/06_best_model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figure saved: 06_best_model_evaluation.png")


# 5. FEATURE IMPORTANCE & SHAP EXPLAINABILITY
print("\n[5/6] Computing feature importance and SHAP values...")

best_model_obj = best['model']

# 5.1 Feature Importance (XGBoost gain) 
# FIX 4: Guard against non-tree models that lack feature_importances_.
#         If Logistic Regression somehow wins the benchmark (edge case on
#         very small datasets), calling .feature_importances_ raises
#         AttributeError. The guard uses coefficient magnitudes instead.

if hasattr(best_model_obj, 'feature_importances_'):
    importances = best_model_obj.feature_importances_
else:
    # Fallback: use absolute logistic regression coefficients
    importances = np.abs(best_model_obj.coef_[0])

importance_df = pd.DataFrame({
    'feature':    FEATURES,
    'importance': importances
}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(
    importance_df['feature'].str.replace('_', ' ').str.title(),
    importance_df['importance'],
    color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance_df)))
)
ax.set_xlabel('Feature Importance (Gain)')
ax.set_title('Top 20 Churn Predictors — XGBoost Feature Importance',
             fontsize=13, fontweight='semibold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('reports/figures/07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figure saved: 07_feature_importance.png")


# 5.2 SHAP Summary Plot 
# FIX 3: Removed plot_size=(10, 8) from shap.summary_plot().
#         That parameter was silently ignored or caused TypeError in SHAP 0.42+.
#         Figure size is now controlled correctly via plt.gcf().set_size_inches()
#         after the plot is created — the correct approach for all SHAP versions.

if SHAP_AVAILABLE:
    try:
        print("  Computing SHAP values on 2,000-record sample...")
        sample_idx = np.random.choice(len(X_test),
                                      size=min(2000, len(X_test)),
                                      replace=False)
        X_shap = X_test.iloc[sample_idx].copy()

        explainer   = shap.TreeExplainer(best_model_obj)
        shap_values = explainer.shap_values(X_shap)

        shap.summary_plot(
            shap_values, X_shap,
            feature_names=[f.replace('_', ' ').title() for f in FEATURES],
            show=False,
            max_display=20
            # plot_size removed ✓ — was deprecated in SHAP 0.42+ causing TypeError
        )
        # Set figure size after SHAP creates the figure
        plt.gcf().set_size_inches(10, 8)
        plt.title('SHAP Feature Impact — Global Churn Drivers',
                  fontsize=13, fontweight='semibold')
        plt.tight_layout()
        plt.savefig('reports/figures/08_shap_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  ✓ Figure saved: 08_shap_summary.png")

    except Exception as e:
        print(f"  ⚠ SHAP plot skipped: {e}")
        print("  → Feature importance chart (07) remains the explainability output.")
else:
    print("  Note: SHAP not available — skipping explainability plot.")


# 6. SCORING PIPELINE — PRODUCTION OUTPUT

print("\n[6/6] Running production scoring pipeline...")

X_full             = df_fe[FEATURES].astype(float)
churn_probabilities = best_model_obj.predict_proba(X_full)[:, 1]


def assign_risk_tier(prob):
    """Classify churn probability into a business action tier."""
    if prob >= 0.70: return 'Critical'
    if prob >= 0.50: return 'High'
    if prob >= 0.30: return 'Medium'
    return 'Low'


scoring_output = df[[
    'customer_id', 'company_size', 'industry', 'contract_type',
    'monthly_charges', 'tenure_months', 'nps_score',
    'num_support_tickets', 'login_frequency_30d'
]].copy()

scoring_output['churn_probability']      = churn_probabilities.round(4)
scoring_output['risk_tier']              = scoring_output['churn_probability'].apply(assign_risk_tier)
scoring_output['revenue_at_risk_monthly'] = (
    scoring_output['monthly_charges'] * scoring_output['churn_probability']
).round(2)
scoring_output['recommended_action'] = scoring_output['risk_tier'].map({
    'Critical': 'Immediate CSM outreach + retention offer within 48h',
    'High':     'Schedule health check call within 1 week',
    'Medium':   'Automated nurture sequence + feature adoption push',
    'Low':      'Standard success cadence — monitor monthly',
})

scoring_output = scoring_output.sort_values('churn_probability', ascending=False)
scoring_output.to_csv('data/processed/churn_scores.csv', index=False)


# Final Summary 

tier_summary = scoring_output.groupby('risk_tier').agg(
    customers=('customer_id', 'count'),
    revenue_at_risk=('revenue_at_risk_monthly', 'sum')
).round(0)

print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"\n  Best Model:        {best_model_name}")
print(f"  AUC-ROC:           {best['auc_roc']:.4f}")
print(f"  Precision (Churn): {best['report']['1']['precision']:.4f}")
print(f"  Recall    (Churn): {best['report']['1']['recall']:.4f}")
print(f"  F1-Score  (Churn): {best['report']['1']['f1-score']:.4f}")
print(f"  Brier Score:       {best['brier']:.4f}")

print(f"\n  RISK TIER BREAKDOWN:")
for tier in ['Critical', 'High', 'Medium', 'Low']:
    if tier in tier_summary.index:
        row = tier_summary.loc[tier]
        print(f"  {tier:<12} {int(row['customers']):>6,} customers   "
              f"${row['revenue_at_risk']:>10,.0f} / month at risk")

total_monthly = scoring_output['revenue_at_risk_monthly'].sum()
print(f"\n  Total Monthly Revenue at Risk:  ${total_monthly:,.0f}")
print(f"  Total Annual  Revenue at Risk:  ${total_monthly * 12:,.0f}")
print(f"\n  Output file: data/processed/churn_scores.csv")
print(f"  Figures:     reports/figures/  (8 charts saved)")
print("\n  ✓ Pipeline complete.")
print("=" * 65)

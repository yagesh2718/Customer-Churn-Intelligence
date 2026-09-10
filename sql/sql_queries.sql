
-- CUSTOMER CHURN INTELLIGENCE PLATFORM — SQL LAYER

-- MODULE 1 — SCHEMA SETUP

-- Drop and recreate for reproducibility (dev environment only)
DROP TABLE IF EXISTS churn_scores CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS product_usage CASCADE;
DROP TABLE IF EXISTS billing_events CASCADE;
DROP TABLE IF EXISTS contracts CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Core customer master
CREATE TABLE customers (
    customer_id         VARCHAR(20)     PRIMARY KEY,
    company_name        VARCHAR(200)    NOT NULL,
    industry            VARCHAR(50),
    company_size        VARCHAR(20)     CHECK (company_size IN ('SMB','Mid-Market','Enterprise')),
    acquisition_date    DATE            NOT NULL,
    csm_owner           VARCHAR(100),
    region              VARCHAR(50),
    created_at          TIMESTAMP       DEFAULT NOW(),
    updated_at          TIMESTAMP       DEFAULT NOW()
);

-- Contracts (one active per customer at any time)
CREATE TABLE contracts (
    contract_id         SERIAL          PRIMARY KEY,
    customer_id         VARCHAR(20)     REFERENCES customers(customer_id),
    contract_type       VARCHAR(20)     CHECK (contract_type IN ('Monthly','Annual','Multi-year')),
    start_date          DATE            NOT NULL,
    end_date            DATE,
    monthly_charges     NUMERIC(12,2)   NOT NULL,
    num_products        SMALLINT        DEFAULT 1,
    has_addon           BOOLEAN         DEFAULT FALSE,
    discount_pct        SMALLINT        DEFAULT 0,
    status              VARCHAR(20)     CHECK (status IN ('Active','Churned','Upgraded','Downgraded')),
    churned_at          TIMESTAMP,
    churn_reason        VARCHAR(100)
);

-- Billing events (payments, delays, refunds)
CREATE TABLE billing_events (
    event_id            SERIAL          PRIMARY KEY,
    customer_id         VARCHAR(20)     REFERENCES customers(customer_id),
    event_date          DATE            NOT NULL,
    event_type          VARCHAR(30)     CHECK (event_type IN ('Payment','Late Payment','Refund','Dispute','Writeoff')),
    amount              NUMERIC(12,2),
    days_overdue        SMALLINT        DEFAULT 0
);

-- Product usage logs (daily granularity)
CREATE TABLE product_usage (
    usage_id            SERIAL          PRIMARY KEY,
    customer_id         VARCHAR(20)     REFERENCES customers(customer_id),
    usage_date          DATE            NOT NULL,
    logins              SMALLINT        DEFAULT 0,
    session_minutes     NUMERIC(8,2)    DEFAULT 0,
    features_used       SMALLINT        DEFAULT 0,
    total_features      SMALLINT        DEFAULT 20,
    api_calls           INT             DEFAULT 0
);

-- Support tickets
CREATE TABLE support_tickets (
    ticket_id           SERIAL          PRIMARY KEY,
    customer_id         VARCHAR(20)     REFERENCES customers(customer_id),
    opened_at           TIMESTAMP       NOT NULL,
    closed_at           TIMESTAMP,
    priority            VARCHAR(10)     CHECK (priority IN ('Low','Medium','High','Critical')),
    category            VARCHAR(50),
    csat_score          SMALLINT        CHECK (csat_score BETWEEN 1 AND 5),
    resolution_days     NUMERIC(6,2)
);

-- NPS survey responses
CREATE TABLE nps_responses (
    response_id         SERIAL          PRIMARY KEY,
    customer_id         VARCHAR(20)     REFERENCES customers(customer_id),
    survey_date         DATE,
    nps_score           SMALLINT        CHECK (nps_score BETWEEN -100 AND 100),
    verbatim_feedback   TEXT
);

-- Churn prediction output (written by Python pipeline)
CREATE TABLE churn_scores (
    score_id            SERIAL          PRIMARY KEY,
    customer_id         VARCHAR(20)     REFERENCES customers(customer_id),
    scored_at           TIMESTAMP       DEFAULT NOW(),
    churn_probability   NUMERIC(6,4)    CHECK (churn_probability BETWEEN 0 AND 1),
    risk_tier           VARCHAR(10)     CHECK (risk_tier IN ('Critical','High','Medium','Low')),
    revenue_at_risk     NUMERIC(12,2),
    top_driver_1        VARCHAR(50),
    top_driver_2        VARCHAR(50),
    top_driver_3        VARCHAR(50),
    recommended_action  TEXT,
    model_version       VARCHAR(20)     DEFAULT 'v1.0'
);

-- Useful indexes
CREATE INDEX idx_contracts_customer   ON contracts(customer_id, status);
CREATE INDEX idx_billing_customer     ON billing_events(customer_id, event_date);
CREATE INDEX idx_usage_customer_date  ON product_usage(customer_id, usage_date);
CREATE INDEX idx_tickets_customer     ON support_tickets(customer_id, opened_at);
CREATE INDEX idx_nps_customer         ON nps_responses(customer_id, survey_date);


-- MODULE 2 — CORE CHURN FEATURE EXTRACTION
-- This CTE chain is the backbone of the ML feature pipeline.
-- Each CTE computes one feature category; the final SELECT joins them all.

WITH

-- ── 2.1 Tenure & contract signals 
tenure_signals AS (
    SELECT
        c.customer_id,
        EXTRACT(MONTH FROM AGE(CURRENT_DATE, c.acquisition_date))::INT     AS tenure_months,
        co.contract_type,
        co.monthly_charges,
        co.num_products,
        co.has_addon::INT                                                   AS has_addon,
        co.discount_pct,
        CASE
            WHEN co.end_date IS NOT NULL
             AND co.end_date <= CURRENT_DATE + INTERVAL '90 days' THEN 1
            ELSE 0
        END                                                                 AS renewal_due_90d,
        c.industry,
        c.company_size,
        c.region,
        CASE WHEN c.csm_owner IS NOT NULL THEN 1 ELSE 0 END               AS csm_assigned
    FROM customers c
    JOIN contracts co
        ON c.customer_id = co.customer_id
       AND co.status = 'Active'
),

-- ── 2.2 Billing & payment risk signals
billing_signals AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (WHERE event_type = 'Late Payment'
                           AND event_date >= CURRENT_DATE - INTERVAL '180 days')
                                                                            AS payment_delays_6m,
        COUNT(*) FILTER (WHERE event_type = 'Dispute'
                           AND event_date >= CURRENT_DATE - INTERVAL '365 days')
                                                                            AS disputes_12m,
        MAX(days_overdue)                                                   AS max_days_overdue,
        SUM(amount) FILTER (WHERE event_type = 'Payment'
                              AND event_date >= CURRENT_DATE - INTERVAL '90 days')
                                                                            AS total_paid_90d
    FROM billing_events
    GROUP BY customer_id
),

-- ── 2.3 Engagement signals (30 / 60 / 90-day windows) 
engagement_signals AS (
    SELECT
        customer_id,
        -- 30-day signals
        AVG(logins)     FILTER (WHERE usage_date >= CURRENT_DATE - 30)     AS login_freq_30d,
        SUM(logins)     FILTER (WHERE usage_date >= CURRENT_DATE - 30)     AS total_logins_30d,
        AVG(session_minutes)
                        FILTER (WHERE usage_date >= CURRENT_DATE - 30)     AS avg_session_min_30d,
        AVG(CASE WHEN total_features > 0
            THEN features_used::FLOAT / total_features ELSE 0 END)
                        FILTER (WHERE usage_date >= CURRENT_DATE - 30)     AS feature_adoption_30d,
        -- 90-day signals
        AVG(logins)     FILTER (WHERE usage_date >= CURRENT_DATE - 90)     AS login_freq_90d,
        -- Recency
        MAX(usage_date)                                                     AS last_active_date,
        CURRENT_DATE - MAX(usage_date)                                      AS days_since_login,
        -- MRR trend: compare last 30d avg logins vs 60-90d avg
        CASE
            WHEN AVG(logins) FILTER (WHERE usage_date BETWEEN
                 CURRENT_DATE - 90 AND CURRENT_DATE - 61) > 0
            THEN (AVG(logins) FILTER (WHERE usage_date >= CURRENT_DATE - 30) -
                  AVG(logins) FILTER (WHERE usage_date BETWEEN
                  CURRENT_DATE - 90 AND CURRENT_DATE - 61))
                / NULLIF(AVG(logins) FILTER (WHERE usage_date BETWEEN
                  CURRENT_DATE - 90 AND CURRENT_DATE - 61), 0)
            ELSE 0
        END                                                                 AS usage_trend_pct
    FROM product_usage
    WHERE usage_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY customer_id
),

-- ── 2.4 Support signals
support_signals AS (
    SELECT
        customer_id,
        COUNT(*)        FILTER (WHERE opened_at >= NOW() - INTERVAL '90 days')
                                                                            AS tickets_90d,
        COUNT(*)        FILTER (WHERE priority IN ('High','Critical')
                                  AND opened_at >= NOW() - INTERVAL '90 days')
                                                                            AS critical_tickets_90d,
        AVG(resolution_days)                                                AS avg_resolution_days,
        AVG(csat_score)                                                     AS avg_csat,
        MAX(csat_score)                                                     AS last_csat,
        COUNT(*) FILTER (WHERE csat_score <= 2
                           AND opened_at >= NOW() - INTERVAL '180 days')   AS low_csat_tickets_6m
    FROM support_tickets
    WHERE opened_at >= NOW() - INTERVAL '365 days'
    GROUP BY customer_id
),

-- ── 2.5 NPS / sentiment signals 
nps_signals AS (
    SELECT DISTINCT ON (customer_id)
        customer_id,
        nps_score,
        survey_date,
        CURRENT_DATE - survey_date                                          AS days_since_nps
    FROM nps_responses
    ORDER BY customer_id, survey_date DESC
),

-- ── 2.6 Revenue history (expansion / contraction / total) 
revenue_signals AS (
    SELECT
        cu.customer_id,
        SUM(CASE WHEN co.status = 'Active' THEN co.monthly_charges ELSE 0 END)
                                                                            AS current_mrr,
        SUM(co.monthly_charges * (
            CASE WHEN co.end_date IS NOT NULL
                 THEN EXTRACT(MONTH FROM AGE(co.end_date, co.start_date))
                 ELSE EXTRACT(MONTH FROM AGE(CURRENT_DATE, co.start_date))
            END
        ))                                                                  AS total_lifetime_revenue,
        COUNT(DISTINCT co.contract_id)                                      AS num_contracts,
        MAX(co.monthly_charges) - MIN(co.monthly_charges)                  AS mrr_expansion
    FROM customers cu
    JOIN contracts co ON cu.customer_id = co.customer_id
    GROUP BY cu.customer_id
),

-- ── 2.7 Composite risk scoring in SQL
risk_scores AS (
    SELECT
        t.customer_id,
        -- Individual risk components (0–1 scale)
        LEAST(COALESCE(e.days_since_login, 90) / 30.0, 3.0) / 3.0        AS recency_risk,
        LEAST(COALESCE(b.payment_delays_6m, 0) * 0.3, 1.0)               AS payment_risk,
        LEAST(COALESCE(s.tickets_90d, 0) *
              COALESCE(s.avg_resolution_days, 1) / 50.0, 1.0)             AS support_risk,
        CASE WHEN n.nps_score IS NOT NULL
             THEN (-n.nps_score / 100.0 + 1.0) / 2.0
             ELSE 0.5 END                                                   AS sentiment_risk
    FROM tenure_signals t
    LEFT JOIN engagement_signals e ON t.customer_id = e.customer_id
    LEFT JOIN billing_signals    b ON t.customer_id = b.customer_id
    LEFT JOIN support_signals    s ON t.customer_id = s.customer_id
    LEFT JOIN nps_signals        n ON t.customer_id = n.customer_id
),

-- ── 2.8 Final feature table join 
final_features AS (
    SELECT
        t.customer_id,

        -- Tenure & contract
        t.tenure_months,
        t.contract_type,
        t.monthly_charges,
        t.num_products,
        t.has_addon,
        t.discount_pct,
        t.renewal_due_90d,
        t.industry,
        t.company_size,
        t.region,
        t.csm_assigned,

        -- Billing
        COALESCE(b.payment_delays_6m, 0)                                   AS payment_delays_6m,
        COALESCE(b.disputes_12m, 0)                                        AS disputes_12m,
        COALESCE(b.max_days_overdue, 0)                                    AS max_days_overdue,

        -- Engagement
        COALESCE(e.login_freq_30d, 0)                                      AS login_frequency_30d,
        COALESCE(e.feature_adoption_30d, 0)                                AS feature_adoption_rate,
        COALESCE(e.avg_session_min_30d, 0)                                 AS avg_session_minutes,
        COALESCE(e.days_since_login, 90)                                   AS last_login_days_ago,
        COALESCE(e.usage_trend_pct, -0.5)                                  AS usage_trend_30d,

        -- Support
        COALESCE(s.tickets_90d, 0)                                         AS num_support_tickets,
        COALESCE(s.avg_resolution_days, 0)                                 AS avg_resolution_days,
        COALESCE(s.avg_csat, 3)                                            AS csat_score,
        COALESCE(s.low_csat_tickets_6m, 0)                                 AS low_csat_tickets_6m,

        -- NPS
        COALESCE(n.nps_score, 0)                                           AS nps_score,
        COALESCE(n.days_since_nps, 365)                                    AS days_since_nps,

        -- Revenue
        COALESCE(rv.total_lifetime_revenue, 0)                             AS total_revenue,
        COALESCE(rv.mrr_expansion, 0)                                      AS mrr_expansion,

        -- Composite risk score
        ROUND(
            (r.recency_risk  * 0.25 +
             r.payment_risk  * 0.25 +
             r.support_risk  * 0.25 +
             r.sentiment_risk * 0.25)::NUMERIC, 4
        )                                                                   AS composite_risk_score

    FROM tenure_signals      t
    LEFT JOIN billing_signals    b  ON t.customer_id = b.customer_id
    LEFT JOIN engagement_signals e  ON t.customer_id = e.customer_id
    LEFT JOIN support_signals    s  ON t.customer_id = s.customer_id
    LEFT JOIN nps_signals        n  ON t.customer_id = n.customer_id
    LEFT JOIN revenue_signals    rv ON t.customer_id = rv.customer_id
    LEFT JOIN risk_scores        r  ON t.customer_id = r.customer_id
)

SELECT * FROM final_features
ORDER BY composite_risk_score DESC;


-- MODULE 3 — COHORT RETENTION ANALYSIS

WITH cohorts AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', acquisition_date)::DATE     AS cohort_month
    FROM customers
),

contract_periods AS (
    SELECT
        c.customer_id,
        co.cohort_month,
        DATE_TRUNC('month', b.event_date)::DATE         AS activity_month
    FROM billing_events b
    JOIN cohorts co ON b.customer_id = co.customer_id
    WHERE b.event_type = 'Payment'
),

retention_grid AS (
    SELECT
        cohort_month,
        EXTRACT(MONTH FROM AGE(activity_month, cohort_month))::INT  AS period_number,
        COUNT(DISTINCT customer_id)                                  AS retained_customers
    FROM contract_periods
    GROUP BY cohort_month, period_number
),

cohort_sizes AS (
    SELECT
        cohort_month,
        COUNT(customer_id)                                           AS cohort_size
    FROM cohorts
    GROUP BY cohort_month
)

SELECT
    rg.cohort_month,
    cs.cohort_size,
    rg.period_number,
    rg.retained_customers,
    ROUND(
        (rg.retained_customers::FLOAT / cs.cohort_size * 100)::NUMERIC,
        1
    )                                                               AS retention_rate_pct
FROM retention_grid rg
JOIN cohort_sizes cs ON rg.cohort_month = cs.cohort_month
ORDER BY cohort_month, period_number;


-- MODULE 4 — REVENUE AT RISK QUANTIFICATION

WITH churn_risk AS (
    SELECT
        cs.customer_id,
        cs.risk_tier,
        cs.churn_probability,
        co.monthly_charges,
        co.monthly_charges * 12                                     AS annual_contract_value,
        co.monthly_charges * cs.churn_probability                   AS expected_monthly_loss,
        co.monthly_charges * 12 * cs.churn_probability              AS expected_annual_loss,
        cu.industry,
        cu.company_size,
        cu.region
    FROM churn_scores cs
    JOIN contracts co ON cs.customer_id = co.customer_id AND co.status = 'Active'
    JOIN customers cu ON cs.customer_id = cu.customer_id
    WHERE cs.scored_at = (
        SELECT MAX(scored_at) FROM churn_scores WHERE customer_id = cs.customer_id
    )
)

SELECT
    risk_tier,
    industry,
    company_size,
    COUNT(*)                                                        AS customers,
    ROUND(AVG(churn_probability)::NUMERIC, 3)                      AS avg_churn_prob,
    ROUND(SUM(monthly_charges)::NUMERIC, 0)                        AS total_mrr,
    ROUND(SUM(expected_monthly_loss)::NUMERIC, 0)                  AS expected_monthly_loss,
    ROUND(SUM(expected_annual_loss)::NUMERIC, 0)                   AS expected_annual_loss,
    ROUND(SUM(annual_contract_value)::NUMERIC, 0)                  AS total_acv
FROM churn_risk
GROUP BY risk_tier, industry, company_size
ORDER BY
    CASE risk_tier WHEN 'Critical' THEN 1 WHEN 'High' THEN 2
                   WHEN 'Medium'   THEN 3 ELSE 4 END,
    expected_monthly_loss DESC;


-- MODULE 5 — EXECUTIVE CHURN KPI DASHBOARD QUERIES

-- ── 5.1 Monthly churn rate trend (24 months) ─────────────────────────────
WITH monthly_starts AS (
    SELECT
        DATE_TRUNC('month', start_date)::DATE       AS month,
        COUNT(*)                                    AS customers_active
    FROM contracts
    WHERE status IN ('Active','Churned')
    GROUP BY DATE_TRUNC('month', start_date)
),
monthly_churns AS (
    SELECT
        DATE_TRUNC('month', churned_at)::DATE       AS month,
        COUNT(*)                                    AS churned,
        SUM(monthly_charges)                        AS churned_mrr
    FROM contracts
    WHERE status = 'Churned'
    GROUP BY DATE_TRUNC('month', churned_at)
)
SELECT
    ms.month,
    ms.customers_active,
    COALESCE(mc.churned, 0)                         AS churned_count,
    COALESCE(mc.churned_mrr, 0)                     AS churned_mrr,
    ROUND(
        (COALESCE(mc.churned, 0)::FLOAT / NULLIF(ms.customers_active, 0) * 100)::NUMERIC,
        2
    )                                               AS monthly_churn_rate_pct
FROM monthly_starts ms
LEFT JOIN monthly_churns mc ON ms.month = mc.month
WHERE ms.month >= CURRENT_DATE - INTERVAL '24 months'
ORDER BY ms.month;


-- ── 5.2 Top 10 accounts by revenue at risk (for Recovery Action Centre) ───
SELECT
    cu.customer_id,
    cu.company_name,
    cu.industry,
    cu.company_size,
    cu.csm_owner,
    co.contract_type,
    co.monthly_charges,
    cs.churn_probability,
    cs.risk_tier,
    cs.revenue_at_risk,
    cs.top_driver_1,
    cs.top_driver_2,
    cs.recommended_action
FROM churn_scores cs
JOIN customers  cu ON cs.customer_id = cu.customer_id
JOIN contracts  co ON cs.customer_id = co.customer_id AND co.status = 'Active'
WHERE cs.risk_tier IN ('Critical', 'High')
  AND cs.scored_at = (
        SELECT MAX(scored_at) FROM churn_scores WHERE customer_id = cs.customer_id
  )
ORDER BY cs.revenue_at_risk DESC
LIMIT 50;


-- ── 5.3 Churn driver frequency (top reasons flagged by model) 
SELECT
    driver,
    COUNT(*) AS customer_count,
    ROUND(AVG(churn_probability)::NUMERIC, 3) AS avg_churn_prob,
    ROUND(SUM(revenue_at_risk)::NUMERIC, 0)   AS total_revenue_at_risk
FROM (
    SELECT customer_id, churn_probability, revenue_at_risk,
           top_driver_1 AS driver FROM churn_scores WHERE top_driver_1 IS NOT NULL
    UNION ALL
    SELECT customer_id, churn_probability, revenue_at_risk,
           top_driver_2 FROM churn_scores WHERE top_driver_2 IS NOT NULL
    UNION ALL
    SELECT customer_id, churn_probability, revenue_at_risk,
           top_driver_3 FROM churn_scores WHERE top_driver_3 IS NOT NULL
) drivers
WHERE scored_at IS NOT NULL  -- ensure latest scores only in production
GROUP BY driver
ORDER BY customer_count DESC;

# Progress Report: Baseline Model Development and Static Conformal Analysis

**Thesis**: Reliability and Interpretability of Uncertainty Estimation in Bus Travel Time Prediction under Temporal Distribution Shifts

**Date**: March 24, 2026 | **Timeline**: Weeks 8–9 (Mar 9 – Mar 22)

**Milestone**: Baseline XGBoost predictor development, static conformal prediction implementation, and Experiment 1 completion

---

## 1. Executive Summary

This report documents the implementation progress for Weeks 8–9, covering the development of the baseline XGBoost travel time prediction model and the first experiment evaluating static conformal prediction under temporal distribution shift.

The implementation follows the thesis methodology precisely: a segment-level bus travel time dataset from Astana, Kazakhstan (785,976 records, 55 days, 3 routes) was preprocessed, features were engineered with strict temporal integrity, and an XGBoost regression model was trained for both route-level and segment-level prediction. The trained model serves as the point predictor upon which conformal prediction intervals are constructed.

**Experiment 1** directly addresses **Research Question 1**: *"How does temporal distribution shift affect the empirical coverage and interval efficiency of conformal prediction for bus ETA?"* The results reveal a critical finding: while static conformal prediction achieves near-perfect calibration on the calibration period (PICP = 0.9007 at 90% target), coverage collapses to approximately 0.61 on all subsequent test periods — a **29 percentage point miscalibration gap**. This confirms the thesis hypothesis that temporal distribution shift undermines the exchangeability assumption underlying conformal prediction, establishing a clear need for adaptive calibration methods (to be investigated in Experiment 2).

---

## 2. Data Preparation

### 2.1 Dataset Description

The study utilizes a publicly available GTFS-based bus trajectory dataset from Astana (Nur-Sultan), Kazakhstan, operated by CTS (City Transportation Systems). The dataset provides segment-level travel time observations — each record represents a single bus traversal of one route segment between consecutive stops, with actual measured run time and dwell time.

| Property | Value |
|----------|-------|
| Total segment records | 785,976 |
| Observation period | July 29 – September 21, 2024 (55 calendar days) |
| Bus routes | 3 (Route 10, Route 12: Railway Station–Airport corridor; Route 46: Karasu–Comfort Town loop) |
| Unique stops | 201 |
| Unique trips | 19,769 |
| Active vehicles | ~52 per day (range: 39–61) |
| Mean segment run time | 106 seconds |
| Mean segment dwell time | 38 seconds |

The choice of this dataset was motivated by its segment-level structure, which is essential for the uncertainty decomposition analysis in Experiment 3, and its sufficient temporal span to study distribution shift.

### 2.2 Data Quality Assessment

A comprehensive quality analysis was performed prior to preprocessing:

- **Completeness**: No null values were found in any of the 785,976 records across all 15 columns, indicating high data collection reliability.
- **Anomalous dates**: September 3 (3,146 records) and September 4 (111 records) were identified as data collection failures — their record counts are 80–99% below the daily norm of 10,000–18,000. These dates were excluded from all subsequent analysis.
- **Outliers**: 925 records had zero run time (physically implausible instantaneous traversals), and extreme values up to 19,140 seconds (5.3 hours for a single segment) were detected. IQR-based outlier removal was applied within each segment-direction group (Q1 − 1.5×IQR, Q3 + 1.5×IQR) to remove these without truncating legitimate travel time variation.
- **Trip completeness**: Trips with fewer than 30 segments were filtered out as incomplete traversals, retaining only full route observations for route-level analysis.

### 2.3 Why Temporal Splitting (Not Random Splitting)

A central methodological decision is the use of **temporal splitting** rather than random train-test splitting. In standard machine learning practice, random splitting ensures that training and test sets are identically distributed (i.i.d. assumption). However, this would mask the very phenomenon we aim to study — temporal distribution shift.

By splitting data chronologically, we deliberately create a scenario where the calibration data (Week 4) may differ distributionally from the test data (Weeks 5–8). This mirrors real-world deployment: a model calibrated on historical data must make predictions on future data whose distribution may have shifted.

| Period | Dates | Days | Segment Records | Trip Records | Purpose |
|--------|-------|------|----------------|-------------|---------|
| **Training** (W1–W3) | Jul 29 – Aug 18 | 21 | ~289,000 | 7,598 | XGBoost model fitting |
| **Calibration** (W4) | Aug 19 – Aug 25 | 7 | ~105,000 | 2,740 | CP nonconformity score computation |
| **Test-Near** (W5) | Aug 26 – Sep 1 | 7 | ~104,000 | 2,707 | Minimal drift evaluation |
| **Test-Mid** (W6) | Sep 2, Sep 5–8 | 5 | ~70,000 | 1,833 | Moderate drift evaluation |
| **Test-Far** (W7–W8) | Sep 9 – Sep 21 | 13 | ~181,000 | 4,736 | Maximum drift evaluation |

The three-tier test structure (Near/Mid/Far) enables systematic measurement of how coverage degrades as a function of temporal distance from calibration — the core investigation of RQ1.

### 2.4 Statistical Evidence of Distribution Shift

To move beyond the assumption that distribution shift exists, we conducted formal statistical tests on route-level travel time distributions across weeks:

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| KS: W3 vs W4 (train→cal boundary) | 0.0521 | 0.0012 | Significant shift at the calibration boundary |
| KS: W7 vs W8 (within test-far) | 0.0494 | 0.0063 | Significant shift even within the test period |
| Kruskal-Wallis (all 8 weeks) | 31.30 | 5.47 × 10⁻⁵ | Overall distribution differs significantly across weeks |

The Kruskal-Wallis test rejects the null hypothesis that all weekly distributions are identical (p < 0.001), providing rigorous statistical evidence that the travel time distribution is **non-stationary** over the 55-day observation period. This validates the experimental premise of RQ1.

---

## 3. Feature Engineering

### 3.1 Design Principles

Features were engineered with two guiding principles:

1. **Temporal integrity**: All historical/statistical features use a strict **past-only lookback window** of 7 days. For a prediction at time *t*, only data from [*t* − 7 days, *t*) is used. This prevents data leakage and ensures that the conformal prediction validity guarantee is not artificially inflated by future information.

2. **Multi-level applicability**: Features were designed to serve both the route-level model (Experiments 1 and 2) and the segment-level model (Experiment 3), with shared temporal and spatial features and level-specific additions.

### 3.2 Feature Set

| Category | Route-Level | Segment-Level | Description |
|----------|------------|---------------|-------------|
| **Temporal** | 8 | 8 | hour_of_day, day_of_week, is_weekend, minute_of_day, cyclical sin/cos encodings for hour (period=24) and day (period=7) |
| **Historical Statistics** | 6 | 6 | 7-day rolling mean, std, median, Q25, Q75, and count of past travel times per (route/segment, direction, time_period) group |
| **Spatial/Route** | 2 | 6 | route_id_encoded, direction_encoded; plus segment_number_normalized, total_route_segments for segment model |
| **Trip Progress** | — | 6 | cumulative_time_so_far, segments_completed, fraction_route_completed, prev_segment run/dwell times (lag-1 to lag-3) |
| **Total** | **16** | **26** | |

### 3.3 Why Historical Features Dominate

The XGBoost feature importance analysis (Section 4.3) reveals that **historical statistical features account for over 56% of total feature importance** at the route level. This is because the best predictor of a bus trip's travel time is the historical average for similar trips (same route, direction, time of day) in the recent past. The 7-day lookback window captures weekly seasonality patterns confirmed by the autocorrelation analysis (significant lag-7 correlation in daily mean travel times).

However, this heavy reliance on historical statistics is precisely what makes the model vulnerable to distribution shift: if traffic conditions change (e.g., roadworks, seasonal demand shifts, weather), the 7-day historical averages will lag behind reality, degrading both point predictions and the conformal prediction intervals built on them.

---

## 4. Baseline XGBoost Model

### 4.1 Why XGBoost

XGBoost was selected as the baseline predictor for several reasons aligned with the thesis methodology:

- **Proven effectiveness**: Tree-based ensembles are well-established for bus ETA prediction, with prior studies demonstrating competitive accuracy versus deep learning in data-constrained settings (refs [17], [23] in the thesis proposal).
- **Computational efficiency**: Training and inference are fast enough for iterative experimentation with conformal prediction, which requires repeated model evaluation.
- **Compatibility**: XGBoost produces deterministic point predictions with a standard scikit-learn API, making it directly compatible with the `calibrated-explanations` conformal prediction framework.
- **Interpretability**: Feature importance is directly available, supporting the thesis's interpretability goals.

The purpose of the baseline model is not to achieve state-of-the-art accuracy, but to provide a **stable predictive foundation** for evaluating uncertainty calibration.

### 4.2 Hyperparameter Tuning

Hyperparameters were tuned using **temporal forward-chaining cross-validation** — a critical methodological choice. Standard random k-fold cross-validation would shuffle temporal order and allow future data to leak into training, invalidating the temporal split design. Instead:

- **Fold 1**: Train on W1, validate on W2
- **Fold 2**: Train on W1–W2, validate on W3

This preserves temporal order and simulates the real-world scenario where models are trained on past data and evaluated on future data.

**Final route-level model configuration**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 200 | Sufficient trees with learning_rate=0.05 |
| max_depth | 8 | Deep enough to capture feature interactions |
| learning_rate | 0.05 | Moderate shrinkage for regularization |
| subsample | 0.9 | Row subsampling for variance reduction |
| colsample_bytree | 0.8 | Feature subsampling |
| reg_lambda | 5 | L2 regularization to prevent overfitting |

### 4.3 Route-Level Model Performance

| Period | MAE (s) | RMSE (s) | MAPE (%) | n_samples |
|--------|---------|----------|----------|-----------|
| Train (W1–W3) | 553.2 | 834.3 | 11.3% | 7,598 |
| Calibration (W4) | 812.7 | 1,144.9 | 15.7% | 2,740 |
| Test-Near (W5) | 800.4 | 1,170.8 | 16.4% | 2,707 |
| Test-Mid (W6) | 821.0 | 1,172.7 | 17.3% | 1,833 |
| Test-Far (W7–W8) | **844.1** | **1,208.8** | 16.4% | 4,736 |

**Key observations**:

- The training MAE (553s) is substantially lower than test MAE (~800–844s), indicating some overfitting but acceptable generalization.
- **MAE increases by +5.5%** from Test-Near (800s) to Test-Far (844s), providing direct evidence of temporal performance degradation. The model becomes progressively less accurate as it predicts further into the future.
- The daily MAE trend line has a positive slope of **+1.94 seconds/day** (Figure P3.5), confirming gradual drift.

### 4.4 Feature Importance Analysis

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | hist_route_mean | 0.274 | Historical |
| 2 | hist_route_q75 | 0.213 | Historical |
| 3 | hist_route_median | 0.079 | Historical |
| 4 | direction_encoded | 0.053 | Spatial |
| 5 | hour_cos | 0.048 | Temporal |
| 6 | is_weekend | 0.047 | Temporal |

Historical statistical features (mean, Q75, median) collectively account for **56.6%** of total feature importance. This dominance means that the model's predictions are heavily anchored to recent historical averages — a strength when conditions are stable, but a vulnerability when conditions shift.

### 4.5 Residual Analysis

The 4-panel residual analysis (Figure P3.2) reveals important properties of the model's errors:

1. **Heteroscedasticity**: The residuals-vs-predicted plot shows a fan-shaped pattern — predictions in the 3,000–5,000s range have smaller residuals than predictions above 7,000s. This means uncertainty is **not constant** across predictions, motivating adaptive interval widths.

2. **Non-Gaussian residuals**: The histogram shows a right-skewed distribution with heavy tails, and the QQ plot deviates significantly from the theoretical normal line at both extremes. The kurtosis exceeds the Gaussian value of 3, confirming fat tails.

3. **No systematic temporal trend in residuals**: The residuals-over-time plot shows no obvious drift pattern, suggesting the model captures the mean trend adequately; the issue is with the *variance* of residuals changing over time.

These residual properties directly motivate the choice of **conformal prediction** (distribution-free, no Gaussian assumption) over parametric uncertainty methods such as Bayesian frameworks or Kalman filters, as discussed in the thesis Related Works chapter.

---

## 5. Experiment 1: Static Conformal Prediction under Temporal Drift

### 5.1 Experimental Design

Experiment 1 investigates **RQ1**: *"How does temporal distribution shift affect the empirical coverage and interval efficiency of conformal prediction for bus ETA?"*

**Method**: Split conformal prediction (also called inductive conformal prediction) was implemented using the `calibrated-explanations` framework. The procedure is:

1. Train XGBoost on Weeks 1–3 (7,598 route-level trips)
2. Compute nonconformity scores on the calibration set (Week 4, 2,740 trips): `score_i = |y_actual_i − y_predicted_i|`
3. Take the quantile of these scores at the desired confidence level: `q = Quantile(scores, ⌈(n+1)(1−α)⌉/n)`
4. For each test prediction: `interval = [y_pred − q, y_pred + q]`

The conformal prediction guarantee states that if calibration and test data are exchangeable (i.i.d.), the prediction interval will contain the true value with probability ≥ (1 − α). The central question is: **what happens when this exchangeability assumption is violated by temporal drift?**

### 5.2 Results at 90% Confidence

| Period | Days from Cal. | PICP | MPIW (s) | Cal. Error | Winkler Score |
|--------|----------------|------|----------|------------|---------------|
| Calibration (W4) | 0 | **0.9007** | 1,528.3 | 0.0007 | — |
| Test-Near (W5) | 1–7 | **0.6169** | 1,528.3 | 0.2831 | — |
| Test-Mid (W6) | 8–14 | **0.6028** | 1,528.3 | 0.2972 | — |
| Test-Far (W7–W8) | 15–28 | **0.6071** | 1,528.3 | 0.2929 | — |

### 5.3 Multi-Confidence Analysis

| Confidence | Calibration PICP | Test-Near PICP | Test-Mid PICP | Test-Far PICP | MPIW (s) |
|------------|-----------------|----------------|---------------|---------------|----------|
| 80% | **0.8007** | 0.4348 | 0.4119 | 0.4259 | 906.6 |
| 90% | **0.9007** | 0.6169 | 0.6028 | 0.6071 | 1,528.3 |
| 95% | **0.9511** | 0.7573 | 0.7387 | 0.7690 | 2,458.0 |

### 5.4 Conditional Coverage Analysis

The conditional coverage heatmap (Figure P4.6) reveals that miscalibration is not uniform across operating conditions:

| Time of Day | Test-Near PICP | Test-Mid PICP | Test-Far PICP |
|-------------|---------------|---------------|---------------|
| Early morning (5–7h) | 0.752 | 0.729 | 0.741 |
| Morning peak (7–10h) | 0.714 | 0.642 | 0.615 |
| Midday (10–16h) | 0.560 | 0.566 | 0.534 |
| Evening peak (16–19h) | 0.570 | 0.553 | 0.665 |
| Evening (19–22h) | 0.593 | 0.626 | 0.658 |
| Night (22–5h) | 0.250 | 1.000 | 1.000 |

- **Early morning** has the highest coverage (~0.73–0.75), likely because travel times are most predictable with low traffic
- **Midday** has the lowest coverage (~0.53–0.57), suggesting the model struggles most during the variable midday period
- **Night** shows extreme values due to very small sample sizes (coverage of 0.25 or 1.00 based on a handful of observations)

### 5.5 Analysis and Interpretation

#### Finding 1: Near-Perfect Calibration on the Calibration Set

The PICP of 0.9007 on Week 4 (versus the 90% target) confirms that the conformal prediction implementation is correct. The split conformal procedure achieves its theoretical guarantee when calibration and test data come from the same distribution. This serves as a **validity anchor** — any subsequent coverage deviation is attributable to distribution shift, not implementation error.

#### Finding 2: Severe and Immediate Coverage Collapse

The most striking result is the **immediate** drop from 0.90 to ~0.61 when moving from the calibration period to Test-Near — just one week later. The coverage does not degrade gradually with temporal distance; instead, it drops sharply and then remains roughly flat across all three test periods (0.617, 0.603, 0.607). This pattern suggests:

- The distribution shift between W4 (calibration) and W5 (test-near) is already substantial
- The shift is not a slow drift but more closely resembles a **regime change** at the calibration boundary
- Additional temporal distance (W6, W7–W8) does not cause further significant degradation

#### Finding 3: Constant-Width Intervals — A Fundamental Limitation

Static conformal prediction produces prediction intervals of **exactly 1,528.3 seconds** (~25.5 minutes) for every single prediction, regardless of the time of day, route, or prediction difficulty. This constant width is a mathematical property of split CP: the quantile `q` is a single scalar applied symmetrically to all predictions.

This has two implications:
1. **No adaptivity**: A trip predicted with high confidence and a trip predicted with low confidence receive identical uncertainty intervals
2. **Width is determined entirely by the calibration set**: If the calibration distribution does not represent the test distribution, the width will be systematically wrong

#### Finding 4: Miscalibration is Consistent Across Confidence Levels

The calibration error is proportionally similar at all three confidence levels: at 80%, the test PICP is ~0.43 (gap: 37 pp); at 90%, ~0.61 (gap: 29 pp); at 95%, ~0.76 (gap: 19 pp). Higher nominal confidence reduces the absolute gap (because wider intervals still capture more), but the relative miscalibration remains severe. The calibration plot (Figure P4.4) shows that the calibration set (W4) lies perfectly on the diagonal (ideal calibration), while all test periods cluster in the **under-coverage region** — below and to the right of the diagonal.

#### Finding 5: Coverage Varies by Operating Condition

The conditional coverage analysis reveals that miscalibration is worst during **midday** (PICP ~0.53) and **evening peak** (PICP ~0.55–0.57), and least severe during **early morning** (PICP ~0.73–0.75). This suggests that certain traffic regimes shifted more between the calibration and test periods than others. Early morning conditions may be more stable (less traffic variability), while midday and peak hours are more susceptible to exogenous changes.

### 5.6 Answer to Research Question 1

> *How does temporal distribution shift affect the empirical coverage and interval efficiency of conformal prediction for bus ETA?*

Temporal distribution shift causes **severe miscalibration** of static conformal prediction intervals for bus ETA. Specifically:

1. **Coverage collapses by 29 percentage points**: From 0.90 (on-target) during calibration to 0.61 during testing — meaning the 90% intervals contain only 61% of actual travel times.

2. **The collapse is immediate, not gradual**: Coverage drops within the first week after calibration and does not further degrade with additional temporal distance, suggesting a regime-change rather than slow-drift pattern.

3. **Miscalibration is systematic across all confidence levels**: At 80%, 90%, and 95% nominal coverage, the empirical coverage falls short by 37, 29, and 19 percentage points respectively.

4. **Constant-width intervals cannot adapt**: Static CP assigns identical 1,528-second intervals to every prediction, making it unable to differentiate between easy and hard prediction scenarios.

5. **Conditional coverage reveals vulnerable operating regimes**: Midday and evening peak periods suffer the worst coverage (~0.53–0.57), while early morning maintains the best (~0.74).

These findings establish a clear need for **adaptive conformal calibration methods** that can update their nonconformity score distribution as new data becomes available — the focus of Experiment 2 in the upcoming implementation phase.

---

## 6. Current Status and Next Steps

### 6.1 Completed Deliverables (Weeks 8–9)

| Deliverable | Figures | Tables | Status |
|-------------|---------|--------|--------|
| Data exploration and quality analysis | 12 | 3 | ✅ Complete |
| Data preprocessing and temporal split | 5 | 3 | ✅ Complete |
| Feature engineering (route + segment) | 3 | 1 | ✅ Complete |
| Baseline XGBoost model (route + segment) | 6 | 3 | ✅ Complete |
| **Experiment 1: Static CP under drift** | **6** | **4** | **✅ Complete** |
| **Total** | **32** | **14** | |

### 6.2 Planned Work (Weeks 10–13)

| Week | Task | Research Question |
|------|------|-------------------|
| 10–11 | Implement online conformal calibration (expanding + sliding window) | RQ2 |
| 10–11 | Compare online vs static CP under temporal drift | RQ2 |
| 12–13 | Implement segment-level uncertainty decomposition | RQ3 |
| 12–13 | Evaluate whether aggregated segment intervals preserve route-level calibration | RQ3 |

### 6.3 Risks and Mitigation

| Risk | Mitigation |
|------|-----------|
| Online CP may not fully recover 90% coverage | Investigate weighted nonconformity scores and adaptive window sizing |
| Segment-level intervals may be too wide when aggregated | Apply Bonferroni correction and evaluate the coverage-width trade-off |
| Computational cost of sequential CP updates | Batch updates (daily rather than per-sample) to manage overhead |

---

## 7. Thesis Writing Status

| Chapter | Status |
|---------|--------|
| Introduction | Draft complete (from planning report) |
| Related Works | Draft complete (from planning report) |
| Research Methodology | Draft complete; being updated with implementation details |
| Results — Data Description | Ready to write (all EDA figures and tables produced) |
| Results — Baseline Model | Ready to write (all performance metrics documented) |
| Results — Experiment 1 | **Ready to write** (all metrics, figures, and analysis complete) |
| Results — Experiment 2 | Pending (Weeks 10–11) |
| Results — Experiment 3 | Pending (Weeks 12–13) |
| Discussion | Pending (Weeks 14–16) |
| Conclusion | Pending (Week 17) |
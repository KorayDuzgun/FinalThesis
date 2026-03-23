# Project Plan: Uncertainty-Aware Bus ETA Prediction Framework

## Context

This plan implements the master's thesis: **"Reliability and Interpretability of Uncertainty Estimation in Bus Travel Time Prediction under Temporal Distribution Shifts"**. The thesis investigates how conformal prediction (CP) can provide statistically sound uncertainty estimates for bus ETA, how those estimates degrade under temporal drift, whether adaptive (online) CP can maintain calibration, and whether segment-level decomposition can attribute uncertainty to specific route segments.

**Dataset**: Astana (Kazakhstan) bus transit data — 785,976 segment-level records across 55 days (Jul 29 - Sep 21, 2024), 3 routes, 201 stops, ~19,769 trips. Anomalous days: Sep 3 (3,146 records) and Sep 4 (111 records) — data collection failure.

**Key tools**: XGBoost (baseline predictor), `calibrated-explanations` Python library (conformal prediction + interpretability), `crepes` (underlying CP engine).

---

## Project Directory Structure

```
ImplementationV1/
├── segment_level_data.csv                    # (existing) Raw data
├── data/gtfs_data/                           # (existing) GTFS reference data
├── requirements.txt                          # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── data_loading.py                       # Data I/O, GTFS joining, caching
│   ├── preprocessing.py                      # Cleaning, outlier removal, aggregation
│   ├── feature_engineering.py                # Feature creation functions
│   ├── temporal_splits.py                    # Temporal train/cal/test split logic
│   ├── evaluation.py                         # MAE, RMSE, PICP, MPIW, calibration error
│   ├── visualization.py                      # Publication-quality plotting helpers
│   └── conformal.py                          # CP wrappers: static, online, segment-level
├── outputs/
│   ├── figures/                              # Saved figures (.pdf and .png)
│   ├── tables/                               # Exported LaTeX/CSV tables
│   ├── models/                               # Saved XGBoost models
│   └── processed_data/                       # Intermediate parquet files
├── notebooks/
│   ├── Phase0_Data_Exploration.ipynb
│   ├── Phase1_Preprocessing.ipynb
│   ├── Phase2_Feature_Engineering.ipynb
│   ├── Phase3_Baseline_XGBoost.ipynb
│   ├── Phase4_Exp1_Static_CP.ipynb
│   ├── Phase5_Exp2_Online_Adaptive_CP.ipynb
│   ├── Phase6_Exp3_Segment_Decomposition.ipynb
│   └── Phase7_Results_Consolidation.ipynb
└── Project_Plan.md
```

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
xgboost>=2.0
scikit-learn>=1.3
calibrated-explanations>=0.11.0
crepes>=0.8.0
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
jupyter>=1.0
jupyterlab>=4.0
ipywidgets>=8.0
statsmodels>=0.14
pingouin>=0.5
tqdm>=4.65
joblib>=1.3
pyarrow>=12.0
```

---

## Temporal Split Strategy

| Period | Dates | Days | Records (~) | Purpose |
|--------|-------|------|-------------|---------|
| **Train (W1-W3)** | Jul 29 – Aug 18 | 21 | ~300K | XGBoost training |
| **Calibration (W4)** | Aug 19 – Aug 25 | 7 | ~109K | CP nonconformity scores |
| **Test-Near (W5)** | Aug 26 – Sep 1 | 7 | ~104K | Minimal drift eval |
| **Excluded** | Sep 3 – Sep 4 | 2 | ~3.3K | Anomalous (data failure) |
| **Test-Mid (W6)** | Sep 2, Sep 5 – Sep 8 | 5 | ~73K | Moderate drift eval |
| **Test-Far (W7-W8)** | Sep 9 – Sep 21 | 13 | ~189K | Maximum drift eval |

**Rationale**: Progressive temporal distance enables measuring drift effects (RQ1). 3-week training captures all day-of-week patterns. Calibration immediately follows training. Test periods at increasing distance.

---

## Utils Module Specifications

### `utils/data_loading.py`
- `load_segment_data(filepath) -> DataFrame` — parse dates/times, set dtypes
- `load_gtfs_trips(gtfs_dir) -> DataFrame` — trip_id, route_id, direction_id, service_id, start_time, end_time, vehicle_id
- `load_gtfs_stops(gtfs_dir) -> DataFrame` — stop_id, stop_name, stop_lat, stop_lon
- `load_gtfs_stop_times(gtfs_dir) -> DataFrame` — trip_id, arrival_time, departure_time, stop_id, stop_sequence
- `load_gtfs_routes(gtfs_dir) -> DataFrame` — route_id, route_short_name, route_long_name
- `load_gtfs_calendar_dates(gtfs_dir) -> DataFrame`
- `join_segment_with_gtfs(segment_df, trips_df, routes_df) -> DataFrame` — add route_id, route_short_name via trip_id
- `cache_dataframe(df, path)` / `load_cached_dataframe(path)` — parquet I/O

### `utils/preprocessing.py`
- `remove_duplicate_records(df) -> DataFrame`
- `detect_and_remove_outliers(df, column, method='iqr', threshold=3.0) -> (cleaned_df, outliers_df)`
- `filter_incomplete_trips(df, min_segments=30) -> DataFrame`
- `filter_anomalous_dates(df, min_daily_records=5000) -> DataFrame`
- `compute_segment_travel_time(df) -> DataFrame` — total_segment_time = run_time + dwell_time
- `aggregate_to_route_level(df) -> DataFrame` — one row per trip: trip_id, route_id, direction, date, departure_time, total_travel_time_seconds, num_segments
- `get_data_quality_report(df) -> dict`

### `utils/feature_engineering.py`
- `add_temporal_features(df)` — hour, minute_of_day, day_of_week, is_weekend, week_number, time_period (morning_peak/midday/evening_peak/night)
- `add_cyclical_time_features(df)` — sin/cos for hour and day_of_week
- `add_historical_segment_statistics(df, lookback_days=7)` — rolling mean, std, median, Q25, Q75 per (segment, direction, time_period)
- `add_historical_route_statistics(df, lookback_days=7)` — same for route-level
- `add_scheduled_vs_actual_deviation(segment_df, stop_times_df)` — delay from GTFS schedule
- `add_cumulative_trip_features(df)` — cumulative_time_so_far, segments_completed, fraction_route_completed
- `add_preceding_segment_features(df)` — lag 1-3 segment run/dwell times
- `add_route_context_features(segment_df, route_stats_df)` — typical route duration for this hour/day
- `get_feature_names(level='route') -> list` — column name lists

### `utils/temporal_splits.py`
- Week boundary constants (W1-W8 date ranges)
- `get_temporal_split_static(df, train_weeks, cal_weeks, test_weeks) -> (train, cal, test)`
- `get_temporal_split_expanding_window(df, initial_train_weeks, step_size_days) -> [(train, cal, test), ...]`
- `get_sliding_window_splits(df, window_size_days, step_days) -> [(cal_window, test_day), ...]`
- `get_temporal_distance(test_df, cal_end_date) -> Series`
- `label_temporal_period(df) -> DataFrame` — adds 'temporal_period' column

### `utils/evaluation.py`
- `compute_mae`, `compute_rmse`, `compute_mape` — point prediction metrics
- `compute_picp(y_true, lower, upper)` — Prediction Interval Coverage Probability
- `compute_mpiw(lower, upper)` — Mean Prediction Interval Width
- `compute_nmpiw(lower, upper, y_range)` — Normalized MPIW
- `compute_calibration_error(y_true, lower, upper, target=0.90)` — |PICP - target|
- `compute_cwc(y_true, lower, upper, target, eta)` — Coverage Width-based Criterion
- `compute_winkler_score(y_true, lower, upper, alpha)` — interval scoring rule
- `compute_all_metrics(y_true, y_pred, lower, upper, target) -> dict`
- `compute_metrics_by_group(y_true, y_pred, lower, upper, groups, target) -> DataFrame`
- `compute_rolling_coverage(y_true, lower, upper, dates, window) -> DataFrame`

### `utils/visualization.py`
- `set_thesis_style()` — serif font, publication-quality rcParams
- `plot_time_series_with_intervals(dates, y_true, y_pred, lower, upper, ...)`
- `plot_coverage_over_time(daily_coverages, target, ...)`
- `plot_interval_width_distribution(widths, ...)`
- `plot_calibration_comparison_bar(results_df, metric, group_col, ...)`
- `plot_coverage_vs_temporal_distance(distances, coverages, ...)`
- `plot_segment_uncertainty_heatmap(segment_uncertainties, ...)`
- `plot_segment_waterfall(segment_contributions, ...)`
- `plot_feature_importance(names, importances, top_n, ...)`
- `plot_residual_analysis(y_true, y_pred, ...)` — 4-panel: residual vs predicted, histogram, QQ, residual vs time
- `plot_data_distribution_comparison(train_values, test_values, feature_name, ...)`
- `create_summary_table(results, save_path) -> DataFrame` — LaTeX export

### `utils/conformal.py`
- `create_calibrated_explainer(model, X_cal, y_cal, mode='regression') -> CalibratedExplainer`
- `get_static_prediction_intervals(explainer, X_test, confidence=0.90) -> (y_pred, lower, upper)`
- `get_online_prediction_intervals(model, X_stream, y_stream, X_cal_init, y_cal_init, confidence, update_freq, window_size) -> (y_pred, lower, upper, running_coverages)`
- `get_segment_level_intervals(segment_models, segment_explainers, X_test_segments, confidence) -> dict`
- `aggregate_segment_intervals_to_route(segment_intervals, trip_mapping, method='sum') -> (y_pred, lower, upper)`
- `compute_segment_uncertainty_attribution(segment_intervals, trip_id) -> DataFrame`

---

## Notebook Details

---

### Phase 0: Data Exploration & Quality Analysis
**File**: `notebooks/Phase0_Data_Exploration.ipynb`
**Thesis**: Establishes the empirical foundation; characterizes the dataset and identifies natural distribution shifts that motivate the research questions.

#### Sections:

1. **Title & Purpose** — Exploratory analysis of Astana bus transit dataset (55 days). Goals: understand schema, identify quality issues, characterize temporal patterns for distribution shift study (RQ1), validate segment-route structure (RQ3).

2. **Load & Inspect Raw Data** — Load segment_level_data.csv. Display .shape, .dtypes, .head(10), .describe().

3. **Load & Inspect GTFS Data** — Load all 6 GTFS files. Display shapes and samples.

4. **Join Segment with GTFS** — Join via trip_id to add route_id, route_short_name. Display route distribution.

5. **Table T0.1: Data Dimensions Summary**
   | Dataset | Records | Columns | Date Range | Description |

6. **Plot P0.1: Daily Record Counts** — Bar chart by date, color-coded weekday/weekend. Highlights Sep 3-4 anomaly.
   - *Thesis*: "The volume drop on Sep 3-4 represents a data anomaly that will affect CP calibration (RQ1). These dates will be excluded."

7. **Plot P0.2: Records per Route per Day** — Stacked area chart by route.

8. **Missing Value Analysis** — Nulls, zeros in run_time/dwell_time, impossible values.

9. **Table T0.2: Missing & Anomalous Value Summary**
   | Column | Null Count | Zero Count | Negative Count | Notes |

10. **Plot P0.3: Distribution of Segment Run Times** — Histogram + box plot (log-scale x-axis).
    - *Thesis*: "Outliers inflate uncertainty intervals. Proper treatment is essential before applying CP."

11. **Plot P0.4: Distribution of Dwell Times** — Same structure.

12. **Plot P0.5: Run Time by Segment Number** — Box plot per segment, faceted by direction.
    - *Thesis*: "Spatial heterogeneity motivates RQ3: some segments contribute disproportionately to route-level uncertainty."

13. **Plot P0.6: Hourly Travel Time Patterns (Heatmap)** — Mean run_time by hour x day_of_week.
    - *Thesis*: "Time-of-day and day-of-week are key features. Systematic weekday/weekend differences represent within-week distribution shifts."

14. **Plot P0.7: Weekly Distribution Shift Visualization** — Violin plot of route-level travel times by week (W1-W8).
    - *Thesis*: "Directly motivates RQ1. If distributions shift visibly across weeks, static CP calibrated on one week may not provide valid coverage on later weeks."

15. **Statistical Test for Distribution Shift** — Kolmogorov-Smirnov test between consecutive weeks + Kruskal-Wallis across all weeks.

16. **Table T0.3: Distribution Shift Statistical Tests**
    | Comparison | KS Statistic | p-value | Significant? |

17. **Plot P0.8: Trip Completeness Analysis** — Histogram of segments-per-trip.
    - *Thesis*: "Trip completeness is essential for route-level aggregation in Experiment 3."

18. **Plot P0.9: Active Buses Per Day** — Unique device_ids per day.

19. **Plot P0.10: Route Map** — Stop lat/lon scatter, color-coded by route.

20. **Plot P0.11: Autocorrelation of Daily Mean Travel Times** — ACF plot.
    - *Thesis*: "Lag-7 autocorrelation confirms weekly seasonality, motivating historical statistics as features."

21. **Plot P0.12: Scheduled vs Actual Deviation Distribution** — Histogram + KDE.

22. **Summary & Key Findings** — 7 bullet points summarizing data characteristics, anomalies, and thesis motivations.

---

### Phase 1: Data Preprocessing & Preparation
**File**: `notebooks/Phase1_Preprocessing.ipynb`
**Thesis**: Prepares clean, reliable data for modeling. Documents every cleaning decision with statistical justification.

#### Sections:

1. **Title & Purpose** — Clean raw data: remove anomalies, duplicates, outliers. Filter incomplete trips. Join with GTFS. Create segment-level and route-level datasets. Establish temporal split.

2. **Load Raw Data**

3. **Remove Exact Duplicates** — Report count removed.

4. **Remove Anomalous Dates (Sep 3-4)** — Justify: 3,146 and 111 records = data collection failure.

5. **Join with GTFS** — Add route_id, route_short_name via trip_id. Verify join completeness.

6. **Plot P1.1: Pre-Cleaning Run Time Distribution**

7. **Outlier Removal** — Per-segment IQR-based (Q1-1.5*IQR, Q3+1.5*IQR) within each segment+direction group.

8. **Table T1.1: Outlier Removal Summary**
   | Route | Direction | Before | Removed | After | % Removed |

9. **Plot P1.2: Before vs After Outlier Removal** — Overlapping histograms.

10. **Filter Incomplete Trips** — Remove trips with < 30 segments. Report statistics.

11. **Compute Total Segment Time** — total_segment_time = run_time + dwell_time

12. **Create Route-Level Dataset** — Aggregate via `aggregate_to_route_level()`. One row per trip.

13. **Table T1.2: Route-Level Dataset Summary**
    | Statistic | Value | (total trips, per route, mean/std travel time, date range)

14. **Plot P1.3: Route-Level Travel Time by Route & Direction** — Box plots.

15. **Apply Temporal Split** — Train W1-W3 (21d), Cal W4 (7d), Test-Near W5 (7d), Test-Mid W6 (5d), Test-Far W7-W8 (13d).

16. **Table T1.3: Temporal Split Statistics**
    | Split | Date Range | Days | Segment Records | Trip Records | Purpose |

17. **Plot P1.4: Temporal Split Visualization** — Timeline with color-coded bands, daily mean travel time overlaid.

18. **Plot P1.5: Distribution Comparison Across Splits** — Overlapping KDEs for train/cal/test_near/test_mid/test_far.
    - *Thesis*: "Visual confirmation of distribution shift: test-far deviates from training distribution, validating RQ1."

19. **Save Processed Data** — `segment_cleaned.parquet`, `route_level.parquet`, `temporal_splits_metadata.json`

20. **Summary**

---

### Phase 2: Feature Engineering
**File**: `notebooks/Phase2_Feature_Engineering.ipynb`
**Thesis**: Creates the feature set for XGBoost. Feature quality directly impacts point predictions and CP intervals.

#### Sections:

1. **Title & Purpose** — 4 feature categories: temporal, historical statistics, spatial/route context, schedule deviation. Strict temporal integrity (past-only lookback).

2. **Load Cleaned Data**

3. **Temporal Features** — hour_of_day, minute_of_day, day_of_week, is_weekend, week_number, time_period (early_morning/morning_peak/midday/evening_peak/evening/night)

4. **Cyclical Encodings** — sin/cos for hour (period=24) and day_of_week (period=7)

5. **Historical Statistics (Route-Level)** — 7-day lookback: mean, std, median, Q25, Q75, count per (route, direction, time_period). First week uses global mean as fallback.

6. **Historical Statistics (Segment-Level)** — Same grouped by (segment, direction, time_period).

7. **Temporal Integrity Note** — "All historical features use strict past-only lookback. No future leakage."

8. **Cumulative Trip Features (Segment-Level)** — cumulative_time_so_far, segments_completed, fraction_route_completed, prev 1-3 segment run/dwell times

9. **Route/Spatial Context** — route_id_encoded, direction_encoded, segment_number (normalized 0-1), total_route_segments

10. **Schedule Deviation** — actual - scheduled arrival, cumulative delay

11. **Plot P2.1: Feature Correlation Matrix** — Heatmap with annotated coefficients.

12. **Plot P2.2: Feature-Target Relationships** — 4-panel scatter grid (hour, hist_mean, day_of_week, is_weekend vs travel_time).

13. **Plot P2.3: Historical Feature Quality Over Time** — Dual-axis: hist_mean vs actual by date, showing tracking quality across drift boundary.

14. **Table T2.1: Final Feature Set Summary**
    | Feature | Type | Level | Description | Source |
    (Route-level: ~15 features; Segment-level: ~22 features)

15. **Save Feature-Engineered Data** — `route_features.parquet`, `segment_features.parquet`

16. **Summary**

---

### Phase 3: Baseline XGBoost Model
**File**: `notebooks/Phase3_Baseline_XGBoost.ipynb`
**Thesis**: XGBoost is the baseline point predictor. Its residuals are what CP uses for interval estimation.

#### Sections:

1. **Title & Purpose** — Train route-level and segment-level XGBoost models. Evaluate on temporal splits.

2. **Load Feature Data & Apply Splits**

3. **Define Hyperparameter Search Space**
   ```
   n_estimators: [200, 500, 1000]
   max_depth: [4, 6, 8]
   learning_rate: [0.01, 0.05, 0.1]
   min_child_weight: [3, 5, 10]
   subsample: [0.8, 0.9]
   colsample_bytree: [0.8, 0.9]
   reg_alpha: [0, 0.1]
   reg_lambda: [1, 5]
   ```

4. **Temporal Cross-Validation Tuning** — Forward-chaining (NOT random k-fold): Fold 1 train W1/val W2, Fold 2 train W1-W2/val W3. RandomizedSearchCV with MAE.

5. **Table T3.1: Top 10 Hyperparameter Configurations**
   | Rank | n_estimators | max_depth | learning_rate | ... | CV MAE (s) |

6. **Train Final Route-Level Model** — Best params on full training set W1-W3.

7. **Evaluate on All Periods** — Cal (W4), Test-Near (W5), Test-Mid (W6), Test-Far (W7-W8).

8. **Table T3.2: Route-Level XGBoost Performance**
   | Period | MAE (s) | RMSE (s) | MAPE (%) | n_samples |
   - *Thesis*: "Error increase from test-near to test-far confirms temporal drift."

9. **Plot P3.1: Actual vs Predicted Scatter** — Colored by test period, identity line.

10. **Plot P3.2: Residual Analysis (4-panel)** — Residuals vs predicted, histogram, QQ, residuals over time.
    - *Thesis*: "Heteroscedasticity motivates adaptive prediction intervals."

11. **Plot P3.3: Feature Importance** — Top 20, horizontal bar chart.

12. **Plot P3.4: MAE by Hour of Day** — Bar chart showing peak-hour difficulty.

13. **Plot P3.5: Daily MAE Over Test Period** — Line plot showing error degradation.
    - *Thesis*: "This degradation is the temporal drift signal motivating all three research questions."

14. **Train Segment-Level Model** — Same process for segment run_time.

15. **Table T3.3: Segment-Level XGBoost Performance** — Same structure.

16. **Plot P3.6: Segment MAE by Segment Position** — Bar chart.
    - *Thesis*: "Segment-level difficulty varies spatially, foreshadowing heterogeneous uncertainty in Exp 3."

17. **Save Models** — `route_xgboost_model.json`, `segment_xgboost_model.json`

18. **Summary**

---

### Phase 4: Experiment 1 — Static CP under Temporal Drift
**File**: `notebooks/Phase4_Exp1_Static_CP.ipynb`
**Thesis**: **RQ1** — "How does temporal distribution shift affect empirical coverage and interval efficiency of conformal prediction for bus ETA?"

#### Sections:

1. **Title, RQ1, Experimental Design** — Static CP with CalibratedExplainer on W4 calibration, evaluate on W5/W6/W7-W8 at 80%/90%/95% confidence.

2. **Load Model & Data**

3. **Create CalibratedExplainer** — `CalibratedExplainer(model, X_cal, y_cal, mode='regression')`

4. **Generate Prediction Intervals (90%)** — `ce.explain_factual(X_test)` → extract intervals

5. **Compute Metrics Per Period**

6. **Table T4.1: Static CP Performance (90% target)**
   | Period | Temporal Distance (days) | PICP | MPIW (s) | Cal. Error | Winkler | n |

7. **RQ1 Analysis** — Detailed paragraph: coverage degradation from near to far, MPIW changes, calibration error growth.

8. **Plot P4.1: Coverage Degradation** — Rolling PICP vs days since calibration, 90% target line.

9. **Plot P4.2: Daily PICP and MPIW** — Dual-axis over dates, train/cal/test color bands.

10. **Plot P4.3: Interval Visualization (Sample Trips)** — Two panels: near vs far, 50 predictions with intervals.

11. **Multi-Confidence Analysis (80%, 90%, 95%)**

12. **Table T4.2: Coverage at Multiple Confidence Levels**
    | Confidence | Period | PICP | MPIW | Cal. Error |

13. **Plot P4.4: Empirical vs Nominal Coverage** — Line per period, perfect calibration diagonal.

14. **Plot P4.5: Interval Width Distribution by Period** — Overlapping KDEs.

15. **Conditional Coverage Analysis** — By time-of-day, day-of-week, route.

16. **Table T4.3: Conditional Coverage**
    | Group | Subgroup | PICP | MPIW | n |

17. **Plot P4.6: Conditional Coverage Heatmap** — PICP by time_period x test_week.

18. **Statistical Significance Tests** — Binomial (PICP vs 90%), Chi-squared (across periods).

19. **Table T4.4: Statistical Significance**
    | Comparison | Test | Statistic | p-value | Conclusion |

20. **Experiment 1 Conclusion** — Full paragraph addressing RQ1 with all table/figure references.

---

### Phase 5: Experiment 2 — Online/Adaptive CP vs Static CP
**File**: `notebooks/Phase5_Exp2_Online_Adaptive_CP.ipynb`
**Thesis**: **RQ2** — "To what extent do online conformal methods improve empirical coverage stability and interval efficiency compared to static CP under drift?"

#### Sections:

1. **Title, RQ2, Experimental Design** — 4 variants: Static CP (baseline), Online Expanding Window, Online Sliding Window 7-day, Online Sliding Window 14-day. Same XGBoost model, different calibration update strategies.

2. **Load Model & Data**

3. **Static CP (Reference from Exp 1)**

4. **Online CP — Expanding Window** — Each test day: generate intervals → observe truth → add to calibration set → re-create CalibratedExplainer.

5. **Online CP — Sliding Window (7-day)** — Same but drop oldest day beyond window.

6. **Online CP — Sliding Window (14-day)**

7. **Table T5.1: Overall Comparison (90%)**
   | Method | PICP | MPIW (s) | Cal. Error | Winkler | CWC |

8. **Plot P5.1: Daily Coverage Comparison** — 4 lines + 90% target.

9. **Plot P5.2: Daily Interval Width Comparison** — 4 lines.

10. **Plot P5.3: Coverage-Width Trade-off** — Scatter per method-period, ideal = top-left.

11. **Table T5.2: Performance by Test Period**
    | Method | Period | PICP | MPIW | Cal. Error |

12. **Plot P5.4: Rolling PICP (Window=200)** — Stability visualization.

13. **Table T5.3: Coverage Stability Metrics**
    | Method | Mean Daily PICP | Std Daily PICP | Max |PICP-90%| | Days PICP < 85% |

14. **Plot P5.5: Interval Examples — Static vs Online** — Side-by-side, 50 samples from Test-Far.

15. **Window Size Sensitivity** — Test 3, 5, 7, 10, 14, 21 days.

16. **Plot P5.6: Window Size Sensitivity** — Dual-axis (PICP, MPIW) vs window size.

17. **Computational Cost**

18. **Table T5.4: Computational Cost**
    | Method | Total Time (s) | Time/Prediction (ms) | Final Cal. Set Size |

19. **Statistical Comparison** — Paired Wilcoxon signed-rank test, bootstrap CIs.

20. **Table T5.5: Statistical Significance**
    | Comparison | Wilcoxon Stat | p-value | Mean PICP Diff | 95% CI |

21. **Experiment 2 Conclusion** — Full paragraph addressing RQ2.

---

### Phase 6: Experiment 3 — Segment-Level Uncertainty Decomposition
**File**: `notebooks/Phase6_Exp3_Segment_Decomposition.ipynb`
**Thesis**: **RQ3** — "Can segment-level decomposition support interpretable uncertainty attribution while preserving route-level calibration?"

#### Sections:

1. **Title, RQ3, Experimental Design** — Segment-level CP → aggregate to route-level. Compare with direct route-level CP. Decompose uncertainty to identify high-risk segments. Use calibrated-explanations' interpretability.

2. **Load Segment Model & Data**

3. **Create Segment-Level CalibratedExplainer** — `CalibratedExplainer(segment_model, X_cal_seg, y_cal_seg, mode='regression')`

4. **Generate Segment-Level Intervals** — Per segment per trip.

5. **Aggregate to Route-Level** — route_lower = Σ(segment_lowers), route_upper = Σ(segment_uppers)

6. **Bonferroni Correction** — Per-segment confidence = 1 - α/N for N segments.

7. **Table T6.1: Route-Level Coverage Comparison**
   | Method | PICP | MPIW (s) | Cal. Error | Winkler |
   (Direct route CP vs aggregated sum vs Bonferroni)

8. **RQ3 Part 1 Analysis** — Does decomposition preserve route-level calibration?

9. **Segment Uncertainty Attribution** — Per-segment interval width, fraction of total, ranking.

10. **Plot P6.1: Segment Uncertainty Bar Chart (Example Trip)** — Interval width by segment number, color = fraction.

11. **Plot P6.2: Average Segment Uncertainty Profile** — Mean width ± std by segment position, faceted by direction.

12. **Plot P6.3: Waterfall — Cumulative Uncertainty Build-Up** — Shows where uncertainty jumps.
    - *Thesis*: "If certain segments contribute disproportionately, transit operators can target improvements at those locations."

13. **Plot P6.4: Top 5 vs Bottom 5 Segments** — Side-by-side bars.

14. **Feature Attribution via CalibratedExplainer** — `ce_segment.explain_factual(X_test_seg)` → per-feature contributions.

15. **Plot P6.5: Feature Attribution — High-Uncertainty Segment** — Waterfall of feature contributions.
    - *Thesis*: "WHY a segment has high uncertainty (e.g., large historical variance, peak hour)."

16. **Plot P6.6: Feature Attribution — Low-Uncertainty Segment** — Contrast.

17. **Temporal Analysis of Segment Uncertainty** — How segment uncertainty changes across test periods.

18. **Plot P6.7: Top-5 Segments Width Over Time** — Line per segment, x = date.

19. **Table T6.2: Segment-Level Uncertainty Statistics**
    | Segment | Direction | Mean Width | Std Width | Frac. Route Uncertainty | Rank |

20. **Plot P6.8: Spatial Uncertainty Map** — Stop lat/lon colored by mean interval width.

21. **Route-Level Calibration by Route**

22. **Table T6.3: Aggregated Coverage by Route**
    | Route | Direction | Direct CP PICP | Aggregated PICP | MPIW Direct | MPIW Aggregated |

23. **Experiment 3 Conclusion** — Full paragraph addressing RQ3.

---

### Phase 7: Results Consolidation & Conclusion
**File**: `notebooks/Phase7_Results_Consolidation.ipynb`
**Thesis**: Synthesizes all experiments into coherent thesis narrative.

#### Sections:

1. **Title & Purpose**

2. **Load All Results**

3. **Table T7.1: Master Results Table**
   | Experiment | Method | PICP | MPIW | Cal. Error | Winkler | Key Finding |

4. **Plot P7.1: Unified Coverage Comparison** — Grouped bar chart (groups=test periods, bars=methods), 90% target line. THE single most important thesis visualization.

5. **Plot P7.2: Coverage-Width Pareto Front** — Scatter with method labels, period shapes.

6. **Plot P7.3: Summary Radar Chart** — Multi-metric: PICP, 1/MPIW, 1/Cal.Error, 1/Winkler, Stability.

7. **Research Question Answers** — 2-3 paragraphs each for RQ1, RQ2, RQ3 with specific numbers.

8. **Table T7.2: Thesis Contributions Summary**
   | Contribution | Evidence | Significance |

9. **Limitations** — Single city, 55 days, 3 routes, XGBoost only, no weather/events, Sep 3-4 anomaly.

10. **Future Work** — Multi-city, deep learning comparison, weather data, real-time deployment, segment interventions.

11. **Export All Tables as LaTeX**

12. **Figure Index** — All figures with paths and thesis section references.

13. **Final Summary** — 2-3 paragraph executive summary.

---

## Implementation Order

1. **Step 1**: Create directory structure + requirements.txt
2. **Step 2**: Implement `utils/` modules (all 7 files)
3. **Step 3**: Phase 0 notebook (exploration)
4. **Step 4**: Phase 1 notebook (preprocessing)
5. **Step 5**: Phase 2 notebook (features)
6. **Step 6**: Phase 3 notebook (XGBoost)
7. **Step 7**: Phase 4 notebook (Experiment 1)
8. **Step 8**: Phase 5 notebook (Experiment 2)
9. **Step 9**: Phase 6 notebook (Experiment 3)
10. **Step 10**: Phase 7 notebook (consolidation)

---

## Verification Plan

After each phase, verify:
- **Phase 0**: All plots render, data dimensions match expected counts, KS tests produce valid p-values
- **Phase 1**: No nulls in cleaned data, temporal split sizes sum to total, parquet files saved
- **Phase 2**: No future leakage (historical features use only past data), feature counts match spec
- **Phase 3**: XGBoost trains without error, MAE reasonable (< 300s route-level), models saved as JSON
- **Phase 4**: PICP near 90% on calibration set, calibrated-explanations API works correctly
- **Phase 5**: Online CP updates daily, coverage tracks across test period, all 4 methods compared
- **Phase 6**: Segment intervals sum correctly, Bonferroni tightens intervals, spatial map renders
- **Phase 7**: All tables/figures from prior phases loaded, LaTeX tables exported

End-to-end test: Run all notebooks in sequence (Phase 0 → Phase 7) and verify all outputs exist in `outputs/`.

---

## Key Architectural Decisions

1. **Route-level for Exp 1 & 2, segment-level for Exp 3**: Route-level is what passengers care about. Segment-level decomposition is specifically for interpretability (RQ3).

2. **`calibrated-explanations` usage**: Main entry point is `CalibratedExplainer`. For online CP (Exp 2), re-instantiate with updated calibration data. If API doesn't expose online mode directly, implement update loop with `crepes` for nonconformity scores.

3. **Segment aggregation**: Simple sum provides a lower bound on route coverage. Bonferroni provides conservative upper bound. Both reported.

4. **Temporal integrity**: All rolling features use strict past-only lookback. First week uses global means as fallback.

---

## Critical Files

- `segment_level_data.csv` — Primary dataset (785,976 records)
- `data/gtfs_data/trips.txt` — Trip → route mapping
- `data/gtfs_data/stops.txt` — Stop coordinates for spatial plots
- `data/gtfs_data/stop_times.txt` — Scheduled times for deviation features
- `data/gtfs_data/routes.txt` — Route names (10, 12, 46)

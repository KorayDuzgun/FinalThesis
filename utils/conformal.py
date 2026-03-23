"""
Conformal Prediction Utilities
==============================
Wrappers around `calibrated-explanations` library and manual conformal
prediction implementations for static, online, and segment-level CP.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def create_calibrated_explainer(model, X_cal, y_cal, feature_names=None, **kwargs):
    """Initialize WrapCalibratedExplainer with a fitted model and calibration data.

    Uses the calibrated-explanations library's WrapCalibratedExplainer which
    wraps a pre-fitted model and calibrates it with held-out data.

    Parameters
    ----------
    model : fitted model
        A trained scikit-learn compatible model (e.g., XGBoost)
    X_cal : np.ndarray
        Calibration features
    y_cal : np.ndarray
        Calibration target values
    feature_names : list, optional
        Feature names for interpretability
    **kwargs
        Kept for backward compatibility; ignored.

    Returns
    -------
    WrapCalibratedExplainer
        Calibrated explainer ready for prediction
    """
    from calibrated_explanations import WrapCalibratedExplainer

    # WrapCalibratedExplainer wraps a pre-fitted model
    # mode is inferred automatically from the model, so we don't pass kwargs
    ce = WrapCalibratedExplainer(model)

    # fit() registers the model; calibrate() computes nonconformity scores
    ce.fit(X_cal, y_cal)
    ce.calibrate(X_cal, y_cal, feature_names=feature_names)

    return ce


def _confidence_to_percentiles(confidence: float) -> tuple:
    """Convert a confidence level to low/high percentiles.

    E.g., confidence=0.90 -> (5, 95) for a 90% interval.
    """
    alpha = (1.0 - confidence) / 2.0
    low = alpha * 100
    high = (1.0 - alpha) * 100
    return (low, high)


def get_static_prediction_intervals(
    explainer,
    X_test,
    confidence: float = 0.90
):
    """Generate prediction intervals using static conformal prediction.

    Uses explain_factual for small datasets (with feature explanations).
    For large datasets, use get_fast_prediction_intervals instead.

    Parameters
    ----------
    explainer : WrapCalibratedExplainer
        Calibrated explainer
    X_test : np.ndarray
        Test features
    confidence : float
        Target coverage probability (e.g., 0.90 for 90%)

    Returns
    -------
    tuple
        (y_pred, lower, upper) arrays
    """
    low_pct, high_pct = _confidence_to_percentiles(confidence)

    # Get factual explanations with specified percentile bounds
    explanations = explainer.explain_factual(
        X_test,
        low_high_percentiles=(low_pct, high_pct)
    )

    # Extract predictions and intervals from the explanation object
    y_pred = []
    lower = []
    upper = []

    for exp in explanations:
        prediction = exp.prediction
        y_pred.append(prediction['predict'])
        lower.append(prediction['low'])
        upper.append(prediction['high'])

    return np.asarray(y_pred), np.asarray(lower), np.asarray(upper)


def get_fast_prediction_intervals(
    model,
    X_cal,
    y_cal,
    X_test,
    confidence: float = 0.90
):
    """Fast bulk prediction intervals using split conformal prediction.

    Computes nonconformity scores on calibration set, takes the quantile,
    and applies symmetric intervals to all test predictions. This is O(n)
    and suitable for large datasets (100K+ samples).

    Parameters
    ----------
    model : fitted model
        Trained model with .predict() method
    X_cal : np.ndarray
        Calibration features
    y_cal : np.ndarray
        Calibration target values
    X_test : np.ndarray
        Test features
    confidence : float
        Target coverage probability (e.g., 0.90 for 90%)

    Returns
    -------
    tuple
        (y_pred, lower, upper) arrays
    """
    X_cal = np.asarray(X_cal)
    y_cal = np.asarray(y_cal)
    X_test = np.asarray(X_test)

    # Step 1: Compute nonconformity scores on calibration set
    y_cal_pred = model.predict(X_cal)
    residuals = np.abs(y_cal - y_cal_pred)

    # Step 2: Get the quantile at the desired confidence level
    # For finite-sample validity: quantile at ceil((n+1)*(1-alpha))/n
    n = len(residuals)
    alpha = 1.0 - confidence
    q_level = min(np.ceil((n + 1) * confidence) / n, 1.0)
    q = np.quantile(residuals, q_level)

    # Step 3: Generate predictions and intervals
    y_pred = model.predict(X_test)
    lower = y_pred - q
    upper = y_pred + q

    return y_pred, lower, upper


def get_normalized_prediction_intervals(
    model,
    X_cal,
    y_cal,
    X_test,
    segment_ids_cal=None,
    segment_ids_test=None,
    confidence: float = 0.90,
    min_samples_per_group: int = 30,
    fallback_to_global: bool = True
):
    """Normalized conformal prediction with per-segment adaptive widths.

    Uses Normalized Conformal Prediction (NCP): nonconformity scores are
    divided by a per-segment difficulty estimate (MAD of residuals), so that
    easy-to-predict segments get narrower intervals and hard segments get wider.

    Parameters
    ----------
    model : fitted model
        Trained model with .predict() method
    X_cal : np.ndarray
        Calibration features
    y_cal : np.ndarray
        Calibration target values
    X_test : np.ndarray
        Test features
    segment_ids_cal : np.ndarray, optional
        Segment identifier for each calibration sample (e.g., segment number).
        If None, uses a difficulty model (MAD of residuals from k-NN).
    segment_ids_test : np.ndarray, optional
        Segment identifier for each test sample
    confidence : float
        Target coverage probability
    min_samples_per_group : int
        Minimum calibration samples per segment to use group-specific sigma
    fallback_to_global : bool
        If a segment has too few calibration samples, fall back to global sigma

    Returns
    -------
    tuple
        (y_pred, lower, upper) arrays — widths vary per sample
    """
    X_cal = np.asarray(X_cal)
    y_cal = np.asarray(y_cal)
    X_test = np.asarray(X_test)

    # Step 1: Calibration residuals
    y_cal_pred = model.predict(X_cal)
    cal_residuals = np.abs(y_cal - y_cal_pred)

    # Step 2: Estimate per-segment difficulty (sigma)
    if segment_ids_cal is not None and segment_ids_test is not None:
        segment_ids_cal = np.asarray(segment_ids_cal)
        segment_ids_test = np.asarray(segment_ids_test)

        # Compute MAD (median absolute deviation) of residuals per segment
        unique_segments = np.unique(segment_ids_cal)
        segment_sigma = {}
        global_sigma = np.median(cal_residuals) + 1e-6  # avoid division by zero

        for seg in unique_segments:
            mask = segment_ids_cal == seg
            if mask.sum() >= min_samples_per_group:
                segment_sigma[seg] = np.median(cal_residuals[mask]) + 1e-6
            elif fallback_to_global:
                segment_sigma[seg] = global_sigma

        # Assign sigma to calibration samples
        sigma_cal = np.array([segment_sigma.get(s, global_sigma) for s in segment_ids_cal])

        # Assign sigma to test samples
        sigma_test = np.array([segment_sigma.get(s, global_sigma) for s in segment_ids_test])
    else:
        # Without segment IDs, use model residual magnitude as difficulty proxy
        # Estimate difficulty via absolute prediction value (larger predictions = more uncertainty)
        y_test_pred = model.predict(X_test)

        # Use calibration residuals grouped by prediction magnitude quintiles
        cal_pred_abs = np.abs(y_cal_pred)
        quintiles = np.percentile(cal_pred_abs, [20, 40, 60, 80])

        def get_sigma(pred_val, cal_preds, cal_res):
            bin_idx = np.digitize(pred_val, quintiles)
            mask = np.digitize(cal_preds, quintiles) == bin_idx
            if mask.sum() >= min_samples_per_group:
                return np.median(cal_res[mask]) + 1e-6
            return np.median(cal_res) + 1e-6

        sigma_cal = np.array([get_sigma(p, cal_pred_abs, cal_residuals) for p in cal_pred_abs])
        sigma_test = np.array([get_sigma(p, cal_pred_abs, cal_residuals) for p in np.abs(y_test_pred)])

    # Step 3: Compute normalized nonconformity scores
    normalized_scores = cal_residuals / sigma_cal

    # Step 4: Get quantile of normalized scores
    n = len(normalized_scores)
    alpha = 1.0 - confidence
    q_level = min(np.ceil((n + 1) * confidence) / n, 1.0)
    q = np.quantile(normalized_scores, q_level)

    # Step 5: Generate predictions and adaptive intervals
    y_pred = model.predict(X_test)
    lower = y_pred - q * sigma_test
    upper = y_pred + q * sigma_test

    return y_pred, lower, upper


def get_online_prediction_intervals(
    model,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    X_cal_init: np.ndarray,
    y_cal_init: np.ndarray,
    confidence: float = 0.90,
    update_frequency: int = 1,
    window_size: int = None,
    dates_stream: np.ndarray = None,
    verbose: bool = True
):
    """Online/adaptive conformal prediction with sequential calibration updates.

    After each prediction batch (defined by update_frequency), revealed true
    values are added to the calibration set, and the explainer is re-created.

    Parameters
    ----------
    model : fitted model
        Trained model (not retrained, only calibration updates)
    X_stream : np.ndarray
        Test features in temporal order
    y_stream : np.ndarray
        True test values (revealed after prediction)
    X_cal_init : np.ndarray
        Initial calibration features
    y_cal_init : np.ndarray
        Initial calibration targets
    confidence : float
        Target coverage
    update_frequency : int
        Number of samples between calibration updates.
        If dates_stream is provided, updates happen daily regardless.
    window_size : int, optional
        If None, use expanding window. If int, use sliding window of this many
        recent samples.
    dates_stream : np.ndarray, optional
        Dates for each sample. If provided, updates happen at each new day.
    verbose : bool
        Whether to show progress bar

    Returns
    -------
    tuple
        (y_pred_all, lower_all, upper_all, running_coverages)
    """
    from calibrated_explanations import WrapCalibratedExplainer

    # Convert to numpy arrays to avoid DataFrame indexing issues
    X_stream = np.asarray(X_stream)
    y_stream = np.asarray(y_stream)
    X_cal = np.copy(np.asarray(X_cal_init))
    y_cal = np.copy(np.asarray(y_cal_init))

    y_pred_all = []
    lower_all = []
    upper_all = []
    running_coverages = []

    n_total = len(X_stream)
    low_pct, high_pct = _confidence_to_percentiles(confidence)

    # Determine update points
    if dates_stream is not None:
        unique_dates = sorted(set(dates_stream))
        date_to_indices = {}
        for i, d in enumerate(dates_stream):
            if d not in date_to_indices:
                date_to_indices[d] = []
            date_to_indices[d].append(i)

        iterator = tqdm(unique_dates, desc='Online CP (daily)') if verbose else unique_dates

        for date in iterator:
            indices = date_to_indices[date]
            X_batch = X_stream[indices]
            y_batch = y_stream[indices]

            # Create explainer with current calibration set
            ce = WrapCalibratedExplainer(model)
            ce.fit(X_cal, y_cal)
            ce.calibrate(X_cal, y_cal)

            # Predict with intervals
            explanations = ce.explain_factual(
                X_batch, low_high_percentiles=(low_pct, high_pct)
            )

            for exp in explanations:
                pred = exp.prediction
                y_pred_all.append(pred['predict'])
                lower_all.append(pred['low'])
                upper_all.append(pred['high'])

            # Update calibration set
            X_cal = np.vstack([X_cal, X_batch])
            y_cal = np.concatenate([y_cal, y_batch])

            # Apply sliding window if specified
            if window_size is not None and len(y_cal) > window_size:
                X_cal = X_cal[-window_size:]
                y_cal = y_cal[-window_size:]

            # Track running coverage
            covered = np.array([(yt >= lo and yt <= up)
                               for yt, lo, up in zip(y_pred_all, lower_all, upper_all)])
            running_coverages.append(np.mean(covered))
    else:
        # Update by sample count
        ce = WrapCalibratedExplainer(model)
        ce.fit(X_cal, y_cal)
        ce.calibrate(X_cal, y_cal)

        iterator = tqdm(range(0, n_total, update_frequency),
                       desc='Online CP') if verbose else range(0, n_total, update_frequency)

        for start in iterator:
            end = min(start + update_frequency, n_total)
            X_batch = X_stream[start:end]
            y_batch = y_stream[start:end]

            # Predict with intervals
            explanations = ce.explain_factual(
                X_batch, low_high_percentiles=(low_pct, high_pct)
            )

            for exp in explanations:
                pred = exp.prediction
                y_pred_all.append(pred['predict'])
                lower_all.append(pred['low'])
                upper_all.append(pred['high'])

            # Update calibration set
            X_cal = np.vstack([X_cal, X_batch])
            y_cal = np.concatenate([y_cal, y_batch])

            if window_size is not None and len(y_cal) > window_size:
                X_cal = X_cal[-window_size:]
                y_cal = y_cal[-window_size:]

            # Re-create explainer with updated calibration
            ce = WrapCalibratedExplainer(model)
            ce.fit(X_cal, y_cal)
            ce.calibrate(X_cal, y_cal)

            covered = np.array([(yt >= lo and yt <= up)
                               for yt, lo, up in zip(y_pred_all, lower_all, upper_all)])
            running_coverages.append(np.mean(covered))

    return (
        np.asarray(y_pred_all),
        np.asarray(lower_all),
        np.asarray(upper_all),
        running_coverages
    )


def get_segment_level_intervals(
    explainer,
    X_test_by_segment: dict,
    confidence: float = 0.90
) -> dict:
    """Get prediction intervals per segment.

    Parameters
    ----------
    explainer : WrapCalibratedExplainer
        Calibrated explainer for the segment model
    X_test_by_segment : dict
        segment_id -> X_test array
    confidence : float
        Target coverage

    Returns
    -------
    dict
        segment_id -> (y_pred, lower, upper)
    """
    low_pct, high_pct = _confidence_to_percentiles(confidence)
    results = {}

    for seg_id, X_test in tqdm(X_test_by_segment.items(), desc='Segment intervals'):
        if len(X_test) == 0:
            continue

        explanations = explainer.explain_factual(
            X_test, low_high_percentiles=(low_pct, high_pct)
        )

        y_pred = []
        lower = []
        upper = []
        for exp in explanations:
            pred = exp.prediction
            y_pred.append(pred['predict'])
            lower.append(pred['low'])
            upper.append(pred['high'])

        results[seg_id] = (
            np.asarray(y_pred),
            np.asarray(lower),
            np.asarray(upper)
        )

    return results


def aggregate_segment_intervals_to_route(
    segment_intervals: dict,
    trip_segment_mapping: pd.DataFrame,
    method: str = 'sum'
):
    """Aggregate segment-level intervals to route-level.

    Parameters
    ----------
    segment_intervals : dict
        segment_id -> (y_pred, lower, upper) for each segment record
    trip_segment_mapping : pd.DataFrame
        DataFrame with trip_id, segment columns to map segments to trips
    method : str
        'sum' for simple summation, 'bonferroni' for Bonferroni-corrected

    Returns
    -------
    pd.DataFrame
        Route-level results: trip_id, y_pred_route, lower_route, upper_route
    """
    results = []

    for trip_id, trip_group in trip_segment_mapping.groupby('trip_id'):
        trip_pred = 0.0
        trip_lower = 0.0
        trip_upper = 0.0
        trip_actual = 0.0
        valid = True

        for _, row in trip_group.iterrows():
            seg_id = row['segment']
            if seg_id in segment_intervals:
                y_pred, lower, upper = segment_intervals[seg_id]
                # Find the corresponding index for this trip's segment
                # This needs to be matched by the order in X_test_by_segment
                idx = row.get('seg_test_idx', 0)
                if idx < len(y_pred):
                    trip_pred += y_pred[idx]
                    trip_lower += lower[idx]
                    trip_upper += upper[idx]
                else:
                    valid = False
                    break
            else:
                valid = False
                break

        if valid:
            results.append({
                'trip_id': trip_id,
                'y_pred_route': trip_pred,
                'lower_route': trip_lower,
                'upper_route': trip_upper,
            })

    return pd.DataFrame(results)


def compute_segment_uncertainty_attribution(
    segment_intervals: dict,
    trip_segments: list
) -> pd.DataFrame:
    """Compute each segment's contribution to total route uncertainty.

    Parameters
    ----------
    segment_intervals : dict
        segment_id -> (y_pred, lower, upper)
    trip_segments : list
        Ordered list of segment IDs in the trip

    Returns
    -------
    pd.DataFrame
        segment, width, fraction_of_total columns
    """
    rows = []
    total_width = 0.0

    for seg_id in trip_segments:
        if seg_id in segment_intervals:
            y_pred, lower, upper = segment_intervals[seg_id]
            # Use mean width across all samples for this segment
            width = np.mean(upper - lower)
        else:
            width = 0.0

        total_width += width
        rows.append({'segment': seg_id, 'width': width})

    df = pd.DataFrame(rows)
    df['fraction_of_total'] = df['width'] / total_width if total_width > 0 else 0
    df['cumulative_width'] = df['width'].cumsum()

    return df

"""
fleetsense.monitoring.drift
===========================

Real-time data drift detection for the FleetSense inference API.

This module maintains a rolling window of observed feature values and
compares them against the training-time reference distribution using
Population Stability Index (PSI). When drift exceeds the configured
threshold it:

  1. Emits a structured WARNING log (picked up by log aggregators / alerting).
  2. Increments a Prometheus counter so Grafana/AlertManager can fire.

The reference distribution is bootstrapped from training-set statistics
stored in ``configs/model.yaml`` (or sensible NYC-taxi defaults).
It is intentionally lightweight — no external dependencies beyond
prometheus-client and standard library — so it cannot crash the API.
"""
from __future__ import annotations

import collections
import math
from typing import Sequence

from prometheus_client import Counter

from fleetsense.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus drift metric (Gap 17)
# ---------------------------------------------------------------------------
DRIFT_ALERTS_TOTAL = Counter(
    "fleetsense_drift_alerts_total",
    "Total number of data drift alerts fired by the inference API",
    ["feature", "severity"],
)

# ---------------------------------------------------------------------------
# Reference distribution for trip_distance (from training data, NYC taxi)
# Bucket edges in miles — must match INCOMING_TRIP_DISTANCE_HISTOGRAM in metrics.py
# ---------------------------------------------------------------------------
_REFERENCE_DISTANCE_BUCKETS: tuple[float, ...] = (
    0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0
)
REFERENCE_DISTANCE_BUCKETS = _REFERENCE_DISTANCE_BUCKETS

# Approximate fraction of trips falling into each bucket for NYC yellow cab
# (derived from TLC 2024-2026 data; update after each major retraining)
_REFERENCE_DISTANCE_FRACTIONS: tuple[float, ...] = (
    0.08,  # 0–0.5
    0.14,  # 0.5–1.0
    0.22,  # 1.0–2.0
    0.18,  # 2.0–3.0
    0.17,  # 3.0–5.0
    0.11,  # 5.0–10.0
    0.04,  # 10.0–15.0
    0.03,  # 15.0–20.0
    0.02,  # 20.0–30.0
    0.01,  # 30.0–50.0+
)


_CACHED_REF: tuple[tuple[float, ...], tuple[float, ...]] | None = None
_CACHED_REF_MTIME: float = 0.0


def load_reference_distribution(force_reload: bool = False) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Load dynamic reference distribution from ARTIFACTS_DIR if present, with in-memory caching and mtime validation."""
    global _CACHED_REF, _CACHED_REF_MTIME

    try:
        from fleetsense.core.constants.paths import ARTIFACTS_DIR
        ref_path = ARTIFACTS_DIR / "reference_distribution.json"
        if ref_path.exists():
            mtime = ref_path.stat().st_mtime
            if _CACHED_REF is None or force_reload or mtime > _CACHED_REF_MTIME:
                import json
                with ref_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if "buckets" in data and "fractions" in data:
                    _CACHED_REF = (tuple(data["buckets"]), tuple(data["fractions"]))
                    _CACHED_REF_MTIME = mtime
                    return _CACHED_REF
    except Exception as exc:
        logger.debug("Could not load dynamic reference distribution: %s", exc)

    if _CACHED_REF is not None and not force_reload:
        return _CACHED_REF

    if _CACHED_REF is None:
        _CACHED_REF = (_REFERENCE_DISTANCE_BUCKETS, _REFERENCE_DISTANCE_FRACTIONS)
    return _CACHED_REF


# PSI thresholds (industry standard)
_PSI_WARNING_THRESHOLD: float = 0.10   # slight drift
_PSI_ALERT_THRESHOLD: float = 0.20    # significant drift

# Reference distribution for predicted duration (in minutes)
_REFERENCE_DURATION_BUCKETS: tuple[float, ...] = (
    3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0
)
_REFERENCE_DURATION_FRACTIONS: tuple[float, ...] = (
    0.10,  # 0-3
    0.15,  # 3-5
    0.25,  # 5-10
    0.20,  # 10-15
    0.15,  # 15-20
    0.08,  # 20-30
    0.05,  # 30-45
    0.02,  # 45-60+
)

# Minimum sample size before drift detection is meaningful
_MIN_WINDOW_SIZE: int = 100

# Rolling window of recent trip distances & predicted durations
_distance_window: collections.deque[float] = collections.deque(maxlen=_MIN_WINDOW_SIZE * 5)
_duration_window: collections.deque[float] = collections.deque(maxlen=_MIN_WINDOW_SIZE * 5)
_sample_counter: int = 0


def _get_psi_thresholds() -> tuple[float, float]:
    """Return warning and alert PSI thresholds from settings or defaults."""
    try:
        from fleetsense.core.config import settings
        warn = getattr(settings.model, "drift_warning_threshold", 0.10)
        alert = getattr(settings.model, "drift_alert_threshold", 0.20)
        return float(warn), float(alert)
    except Exception:
        return 0.10, 0.20


def _assign_bucket(value: float, edges: tuple[float, ...]) -> int:
    """Return the bucket index for a value given sorted bucket edges."""
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)  # last bucket (beyond max edge)


def _compute_psi(
    observed_fractions: Sequence[float],
    expected_fractions: Sequence[float],
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions."""
    psi = 0.0
    for obs, exp in zip(observed_fractions, expected_fractions):
        obs = max(obs, epsilon)
        exp = max(exp, epsilon)
        psi += (obs - exp) * math.log(obs / exp)
    return psi


def _evaluate_distribution_drift(
    window: collections.deque[float],
    buckets: tuple[float, ...],
    reference_fractions: tuple[float, ...],
    feature_name: str,
) -> float:
    """Compute PSI for a rolling window and alert if thresholds are exceeded."""
    floor = 0.005  # 0.5% baseline smoothing floor per bucket
    n_buckets = len(buckets) + 1
    observed_counts = [0] * n_buckets
    for v in window:
        idx = _assign_bucket(v, buckets)
        observed_counts[idx] += 1

    total = sum(observed_counts)
    observed_fractions = [max(c / total, floor) for c in observed_counts]
    obs_sum = sum(observed_fractions)
    observed_fractions = [f / obs_sum for f in observed_fractions]

    ref_fracs = list(reference_fractions)
    while len(ref_fracs) < n_buckets:
        ref_fracs.append(floor)
    ref_fracs = [max(f, floor) for f in ref_fracs]
    ref_total = sum(ref_fracs)
    ref_fracs = [f / ref_total for f in ref_fracs]

    psi = _compute_psi(observed_fractions, ref_fracs, epsilon=1e-4)
    warning_threshold, alert_threshold = _get_psi_thresholds()

    if psi >= alert_threshold:
        logger.warning(
            "DATA_DRIFT_ALERT feature=%s psi=%.4f threshold=%.2f severity=HIGH "
            "window_size=%d — Significant distribution shift detected. "
            "Consider retraining the model.",
            feature_name,
            psi,
            alert_threshold,
            len(window),
        )
        DRIFT_ALERTS_TOTAL.labels(feature=feature_name, severity="HIGH").inc()
    elif psi >= warning_threshold:
        logger.warning(
            "DATA_DRIFT_ALERT feature=%s psi=%.4f threshold=%.2f severity=MEDIUM "
            "window_size=%d — Moderate distribution shift detected.",
            feature_name,
            psi,
            warning_threshold,
            len(window),
        )
        DRIFT_ALERTS_TOTAL.labels(feature=feature_name, severity="MEDIUM").inc()

    return psi


def check_and_alert_drift(
    trip_distance: float,
    predicted_duration_minutes: float | None = None,
) -> None:
    """
    Record latest observations and check for distribution drift in input features
    and predicted durations.
    """
    global _sample_counter
    _distance_window.append(trip_distance)
    if predicted_duration_minutes is not None:
        _duration_window.append(predicted_duration_minutes)
    _sample_counter += 1

    if len(_distance_window) < _MIN_WINDOW_SIZE:
        return

    # Throttle: Evaluate PSI every 10 samples once window is populated
    if _sample_counter % 10 != 0:
        return

    dist_buckets, dist_ref_fractions = load_reference_distribution()
    _evaluate_distribution_drift(
        window=_distance_window,
        buckets=dist_buckets,
        reference_fractions=dist_ref_fractions,
        feature_name="trip_distance",
    )

    if len(_duration_window) >= _MIN_WINDOW_SIZE:
        _evaluate_distribution_drift(
            window=_duration_window,
            buckets=_REFERENCE_DURATION_BUCKETS,
            reference_fractions=_REFERENCE_DURATION_FRACTIONS,
            feature_name="predicted_duration_minutes",
        )

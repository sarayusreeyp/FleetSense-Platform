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


def load_reference_distribution() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Load dynamic reference distribution from ARTIFACTS_DIR if present, else fallback."""
    try:
        from fleetsense.core.constants.paths import ARTIFACTS_DIR
        ref_path = ARTIFACTS_DIR / "reference_distribution.json"
        if ref_path.exists():
            import json
            with ref_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "buckets" in data and "fractions" in data:
                return tuple(data["buckets"]), tuple(data["fractions"])
    except Exception as exc:
        logger.debug("Could not load dynamic reference distribution: %s", exc)
    return _REFERENCE_DISTANCE_BUCKETS, _REFERENCE_DISTANCE_FRACTIONS


# PSI thresholds (industry standard)
_PSI_WARNING_THRESHOLD: float = 0.10   # slight drift
_PSI_ALERT_THRESHOLD: float = 0.20    # significant drift

# Minimum sample size before drift detection is meaningful
_MIN_WINDOW_SIZE: int = 100

# Rolling window of recent trip distances
_distance_window: collections.deque[float] = collections.deque(maxlen=_MIN_WINDOW_SIZE * 5)


def _assign_bucket(value: float, edges: tuple[float, ...]) -> int:
    """Return the bucket index for a value given sorted bucket edges."""
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)  # last bucket (beyond max edge)


def _compute_psi(
    observed_fractions: Sequence[float],
    expected_fractions: Sequence[float],
    epsilon: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions."""
    psi = 0.0
    for obs, exp in zip(observed_fractions, expected_fractions):
        obs = max(obs, epsilon)
        exp = max(exp, epsilon)
        psi += (obs - exp) * math.log(obs / exp)
    return psi


def check_and_alert_drift(trip_distance: float) -> None:
    """
    Record the latest trip_distance observation and check for distribution drift.

    Called on every prediction — designed to be very fast (O(n) only when
    window is full enough to evaluate, otherwise O(1) deque append).
    """
    _distance_window.append(trip_distance)

    if len(_distance_window) < _MIN_WINDOW_SIZE:
        return  # not enough data yet

    # Throttle: Evaluate PSI every 10 samples to avoid excessive CPU overhead on high-throughput paths
    if len(_distance_window) % 10 != 0 and len(_distance_window) != _distance_window.maxlen:
        return

    buckets, reference_fractions = load_reference_distribution()

    # Build observed bucket distribution from the rolling window
    n_buckets = len(buckets) + 1
    observed_counts = [0] * n_buckets
    for d in _distance_window:
        idx = _assign_bucket(d, buckets)
        observed_counts[idx] += 1

    total = sum(observed_counts)
    observed_fractions = [c / total for c in observed_counts]

    # Pad reference fractions to match n_buckets (last bucket absorbs overflow)
    ref_fracs = list(reference_fractions)
    while len(ref_fracs) < n_buckets:
        ref_fracs.append(1e-6)
    # Normalize to ensure they sum to 1
    ref_total = sum(ref_fracs)
    ref_fracs = [f / ref_total for f in ref_fracs]

    psi = _compute_psi(observed_fractions, ref_fracs)

    if psi >= _PSI_ALERT_THRESHOLD:
        logger.warning(
            "DATA_DRIFT_ALERT feature=trip_distance psi=%.4f threshold=%.2f severity=HIGH "
            "window_size=%d — Significant distribution shift detected. "
            "Consider retraining the model.",
            psi,
            _PSI_ALERT_THRESHOLD,
            len(_distance_window),
        )
        DRIFT_ALERTS_TOTAL.labels(feature="trip_distance", severity="HIGH").inc()
    elif psi >= _PSI_WARNING_THRESHOLD:
        logger.warning(
            "DATA_DRIFT_ALERT feature=trip_distance psi=%.4f threshold=%.2f severity=MEDIUM "
            "window_size=%d — Moderate distribution shift detected.",
            psi,
            _PSI_WARNING_THRESHOLD,
            len(_distance_window),
        )
        DRIFT_ALERTS_TOTAL.labels(feature="trip_distance", severity="MEDIUM").inc()

from fleetsense.monitoring.metrics import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    INCOMING_TRIP_DISTANCE_HISTOGRAM,
    MODEL_LOAD_STATUS,
    PREDICTED_DURATION_HISTOGRAM,
    PREDICTIONS_TOTAL,
    get_latest_metrics,
    record_batch_predictions,
    record_prediction,
    set_model_status,
)

__all__ = [
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION_SECONDS",
    "PREDICTIONS_TOTAL",
    "PREDICTED_DURATION_HISTOGRAM",
    "INCOMING_TRIP_DISTANCE_HISTOGRAM",
    "MODEL_LOAD_STATUS",
    "record_prediction",
    "record_batch_predictions",
    "set_model_status",
    "get_latest_metrics",
]

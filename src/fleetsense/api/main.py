from __future__ import annotations

import collections
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from fleetsense.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TripFeaturesPayload,
)
from fleetsense.api.service import PredictionService
from fleetsense.core.config import settings
from fleetsense.core.exceptions import FleetSenseError
from fleetsense.core.logger import get_logger
from fleetsense.monitoring import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    check_and_alert_batch_drift,
    check_and_alert_drift,
    get_latest_metrics,
    record_batch_predictions,
    record_prediction,
    set_model_status,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Service initialization lock to prevent duplicate parallel downloads during cold starts
_service_init_lock = threading.Lock()


def get_prediction_service() -> PredictionService:
    """Thread-safe singleton accessor for PredictionService."""
    service = getattr(app.state, "prediction_service", None)
    if service is None:
        with _service_init_lock:
            service = getattr(app.state, "prediction_service", None)
            if service is None:
                service = PredictionService()
                app.state.prediction_service = service
    return service


# ---------------------------------------------------------------------------
# Rate Limiter (Gap 19 - zero third-party dependency sliding window)
# ---------------------------------------------------------------------------
class InMemoryRateLimiter:
    """Sliding-window rate limiter per client IP without external dependencies."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.rate = requests_per_minute
        self.window = 60.0  # seconds
        self._history: dict[str, collections.deque[float]] = {}
        self._last_cleanup = time.time()

    def check(self, client_ip: str) -> bool:
        now = time.time()
        # Periodic cleanup of idle client IPs every 60 seconds to prevent unbounded memory growth
        if now - self._last_cleanup > 60.0:
            stale_cutoff = now - self.window
            stale_keys = [ip for ip, dq in self._history.items() if not dq or dq[-1] < stale_cutoff]
            for ip in stale_keys:
                del self._history[ip]
            self._last_cleanup = now

        if client_ip not in self._history:
            self._history[client_ip] = collections.deque()
        timestamps = self._history[client_ip]

        # Drop timestamps older than 60 seconds
        while timestamps and timestamps[0] < now - self.window:
            timestamps.popleft()
        if len(timestamps) >= self.rate:
            return False
        timestamps.append(now)
        return True


rate_limiter_single = InMemoryRateLimiter(requests_per_minute=60)
rate_limiter_batch = InMemoryRateLimiter(requests_per_minute=20)


def _get_client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Forwarded-For behind reverse proxies / ingress."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Leftmost entry is the client's original IP
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# /metrics API Key security (Gap 21)
# ---------------------------------------------------------------------------
METRICS_API_KEY = os.getenv("FLEETSENSE_METRICS_API_KEY") or os.getenv("METRICS_API_KEY") or ""
_metrics_api_key_header = APIKeyHeader(name="X-Metrics-API-Key", auto_error=False)


def _require_metrics_key(
    request: Request,
    api_key: str | None = Depends(_metrics_api_key_header),
) -> None:
    """Require a secret API key (X-Metrics-API-Key or Bearer token) to access /metrics if configured."""
    if not METRICS_API_KEY:
        return
    auth_header = request.headers.get("Authorization", "")
    bearer_token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else None
    if api_key == METRICS_API_KEY or bearer_token == METRICS_API_KEY:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Valid X-Metrics-API-Key or Authorization Bearer token required to access /metrics.",
    )


# ---------------------------------------------------------------------------
# Allowed CORS origins (Gap 20)
# ---------------------------------------------------------------------------
def _get_allowed_origins() -> list[str]:
    """Return allowed CORS origins from env var; defaults to localhost + common dev/prod origins."""
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    env = settings.app.environment
    # Allow local frontend ports by default, and log guidance for custom prod domains
    default_origins = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]
    if env in ("production", "staging"):
        logger.info(
            "CORS_ALLOWED_ORIGINS not explicitly configured in %s; using default dashboard origins %s. "
            "Set CORS_ALLOWED_ORIGINS=https://yourdomain.com for custom domains.",
            env,
            default_origins,
        )
    return default_origins


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for loading models upon startup."""
    logger.info("Initializing FleetSense Prediction Service...")
    service = PredictionService()
    app.state.prediction_service = service
    set_model_status(
        loaded=(service.model is not None),
        model_name=service.algorithm or "None",
    )
    if service.is_dummy:
        logger.warning(
            "WARNING: Running on DUMMY baseline model (source=%s). "
            "This is only safe in testing environments. "
            "Train a real model and deploy it to production.",
            service.model_source,
        )
    else:
        logger.info("Real trained model loaded from: %s", service.model_source)
    yield
    logger.info("FleetSense Prediction Service shutdown.")


app = FastAPI(
    title="FleetSense ML Inference API",
    description="Production-grade real-time taxi trip duration and demand forecasting API.",
    version=settings.app.version,
    lifespan=lifespan,
)

# Restrict CORS (Gap 20 & W3C compliance)
_cors_origins = _get_allowed_origins()
_has_wildcard_origin = "*" in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _has_wildcard_origin,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Prometheus middleware for HTTP request rate and latency metrics."""
    # Attach a unique request ID for audit trail correlation
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()
    raw_path = request.url.path
    known_endpoints = {"/", "/health", "/metrics", "/predict", "/predict/batch", "/docs", "/openapi.json"}
    endpoint = raw_path if raw_path in known_endpoints else "unmatched"

    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        duration = time.perf_counter() - start_time
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status="500",
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)
        raise


@app.exception_handler(FleetSenseError)
async def fleetsense_error_handler(request: Request, exc: FleetSenseError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.get(
    "/",
    tags=["Root"],
    summary="Root Status",
)
async def root() -> dict[str, str]:
    return {
        "service": "FleetSense ML Platform",
        "version": settings.app.version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
)
async def health_check() -> HealthResponse:
    """Return system status, environment, and model loading state."""
    service: PredictionService = getattr(app.state, "prediction_service", None)
    model_loaded = service is not None and service.model is not None
    algorithm = service.algorithm if service else None
    model_source = service.model_source if service else None
    is_dummy = service.is_dummy if service else False

    # Gap 23: degrade status if the model is a dummy baseline
    health_status = "healthy"
    if not model_loaded:
        health_status = "unavailable"
    elif is_dummy:
        health_status = "degraded"

    return HealthResponse(
        status=health_status,
        app_name=settings.app.name,
        environment=settings.app.environment,
        version=settings.app.version,
        model_loaded=model_loaded,
        model_algorithm=algorithm,
        model_source=model_source,
        is_dummy_model=is_dummy,
        mlflow_run_id=getattr(service, "mlflow_run_id", None) if service else None,
        mlflow_model_version=getattr(service, "mlflow_model_version", None) if service else None,
    )


@app.post(
    "/model/reload",
    tags=["Management"],
    summary="Hot-Reload Active Model from MLflow Registry",
    dependencies=[Depends(_require_metrics_key)],
)
async def reload_active_model(request: Request) -> dict[str, Any]:
    """
    Trigger dynamic hot-reloading of the ML model from the MLflow Model Registry
    without restarting the API server. Protected by admin/metrics API key.
    """
    service = get_prediction_service()

    reload_result = service.reload_model()
    set_model_status(
        loaded=(service.model is not None),
        model_name=service.algorithm or "None",
    )
    logger.info("MODEL_RELOAD_EVENT result=%s", reload_result)
    return {
        "status": "success" if service.model is not None else "failed",
        "details": reload_result,
        "mlflow_run_id": service.mlflow_run_id,
        "mlflow_model_version": service.mlflow_model_version,
    }


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Prometheus Telemetry Endpoint",
    dependencies=[Depends(_require_metrics_key)],
)
async def metrics(request: Request) -> Response:
    """Expose Prometheus metrics for scraping by Prometheus / OpenTelemetry."""
    content, media_type = get_latest_metrics()
    return Response(content=content, media_type=media_type)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Single Trip Duration Prediction",
)
async def predict_trip(
    request: Request,
    payload: TripFeaturesPayload,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    """
    Predict the duration of an individual taxi trip based on pickup time,
    distance, and location zones.
    """
    # Rate limiting check
    client_ip = _get_client_ip(request)
    if not rate_limiter_single.check(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 60 requests per minute.",
        )

    service = get_prediction_service()

    if service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not currently loaded on the server.",
        )

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    try:
        result = service.predict_trip(payload)

        # Gap 16: structured prediction audit trail
        logger.info(
            "PREDICTION_AUDIT request_id=%s vendor=%s distance=%.2f pu=%d do=%d "
            "pickup=%s predicted_minutes=%.2f model_version=%s model_source=%s",
            request_id,
            payload.VendorID,
            payload.trip_distance,
            payload.PULocationID,
            payload.DOLocationID,
            payload.pickup_datetime.isoformat(),
            result.predicted_duration_minutes,
            result.model_version,
            result.model_source,
        )

        record_prediction(
            predicted_duration_minutes=result.predicted_duration_minutes,
            trip_distance=payload.trip_distance,
            model_version=result.model_version,
            status="success",
        )

        # Offload drift check to background tasks to keep prediction latency minimal
        background_tasks.add_task(
            check_and_alert_drift,
            trip_distance=payload.trip_distance,
            predicted_duration_minutes=result.predicted_duration_minutes,
        )

        return result
    except HTTPException:
        raise
    except FleetSenseError:
        raise
    except Exception as exc:
        record_prediction(
            predicted_duration_minutes=0.0,
            trip_distance=payload.trip_distance,
            status="error",
        )
        logger.error(
            "PREDICTION_AUDIT request_id=%s status=error error=%s",
            request_id,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "API-500",
                "message": "Inference error occurred while processing the request.",
                "details": {"error": str(exc)},
            },
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch Trip Duration Predictions",
)
async def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
) -> BatchPredictionResponse:
    """
    Predict durations for multiple taxi trips in a single vectorized batch pass.
    """
    # Rate limiting check
    client_ip = _get_client_ip(request)
    if not rate_limiter_batch.check(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 20 batch requests per minute.",
        )

    service = get_prediction_service()

    if service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not currently loaded on the server.",
        )

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    try:
        result = service.predict_batch(batch_request)
        durations = [p.predicted_duration_minutes for p in result.predictions]
        distances = [p.trip_distance for p in batch_request.trips]
        model_version = result.predictions[0].model_version if result.predictions else "v1"
        model_source = result.predictions[0].model_source if result.predictions else "unknown"

        # Gap 16: batch audit trail
        logger.info(
            "BATCH_PREDICTION_AUDIT request_id=%s trips=%d model_version=%s model_source=%s "
            "avg_predicted_minutes=%.2f",
            request_id,
            len(result.predictions),
            model_version,
            model_source,
            sum(durations) / len(durations) if durations else 0.0,
        )

        record_batch_predictions(
            durations=durations,
            distances=distances,
            model_version=model_version,
            status="success",
        )

        # Offload batch drift check as a single background task (zero event loop lag)
        background_tasks.add_task(
            check_and_alert_batch_drift,
            distances=distances,
            predicted_durations=durations,
        )

        return result
    except HTTPException:
        raise
    except FleetSenseError:
        raise
    except Exception as exc:
        record_batch_predictions(
            durations=[0.0] * len(batch_request.trips),
            distances=[p.trip_distance for p in batch_request.trips],
            status="error",
        )
        logger.error(
            "BATCH_PREDICTION_AUDIT request_id=%s status=error error=%s",
            request_id,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "API-500",
                "message": "Batch inference error occurred while processing the request.",
                "details": {"error": str(exc)},
            },
        )

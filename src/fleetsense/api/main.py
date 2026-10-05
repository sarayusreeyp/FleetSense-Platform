from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fleetsense.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TripFeaturesPayload,
)
from fleetsense.api.service import PredictionService
from fleetsense.core.config import settings
from fleetsense.core.exceptions import DataValidationError, FleetSenseError
from fleetsense.core.logger import get_logger
from fleetsense.monitoring import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    get_latest_metrics,
    record_batch_predictions,
    record_prediction,
    set_model_status,
)

logger = get_logger(__name__)


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
    yield
    logger.info("FleetSense Prediction Service shutdown.")


app = FastAPI(
    title="FleetSense ML Inference API",
    description="Production-grade real-time taxi trip duration and demand forecasting API.",
    version=settings.app.version,
    lifespan=lifespan,
)

# Enable CORS for frontend clients / dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Prometheus middleware for HTTP request rate and latency metrics."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    endpoint = request.url.path

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=endpoint,
        status=str(response.status_code),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        endpoint=endpoint,
    ).observe(duration)

    return response


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

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        app_name=settings.app.name,
        environment=settings.app.environment,
        version=settings.app.version,
        model_loaded=model_loaded,
        model_algorithm=algorithm,
    )


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Prometheus Telemetry Endpoint",
)
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping by Prometheus / OpenTelemetry."""
    content, media_type = get_latest_metrics()
    return Response(content=content, media_type=media_type)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Single Trip Duration Prediction",
)
async def predict_trip(payload: TripFeaturesPayload) -> PredictionResponse:
    """
    Predict the duration of an individual taxi trip based on pickup time,
    distance, and location zones.
    """
    service: PredictionService = getattr(app.state, "prediction_service", None)
    if not service:
        service = PredictionService()
        app.state.prediction_service = service

    try:
        result = service.predict_trip(payload)
        record_prediction(
            predicted_duration_minutes=result.predicted_duration_minutes,
            trip_distance=payload.trip_distance,
            model_version=result.model_version,
            status="success",
        )
        return result
    except Exception as exc:
        record_prediction(
            predicted_duration_minutes=0.0,
            trip_distance=payload.trip_distance,
            status="error",
        )
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch Trip Duration Predictions",
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict durations for multiple taxi trips in a single vectorized batch pass.
    """
    service: PredictionService = getattr(app.state, "prediction_service", None)
    if not service:
        service = PredictionService()
        app.state.prediction_service = service

    try:
        result = service.predict_batch(request)
        durations = [p.predicted_duration_minutes for p in result.predictions]
        distances = [p.trip_distance for p in request.trips]
        model_version = result.predictions[0].model_version if result.predictions else "v1"
        record_batch_predictions(
            durations=durations,
            distances=distances,
            model_version=model_version,
            status="success",
        )
        return result
    except Exception as exc:
        record_batch_predictions(
            durations=[0.0] * len(request.trips),
            distances=[p.trip_distance for p in request.trips],
            status="error",
        )
        logger.error("Batch inference failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {exc}",
        )

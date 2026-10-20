from typing import Any

from .base import FleetSenseError


class ModelTrainingError(FleetSenseError):

    def __init__(
        self,
        message: str = "Model training failed.",
        details: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(
            message=message,
            error_code="ML-001",
            details=details,
        )


class FeatureEngineeringError(FleetSenseError):

    def __init__(
        self,
        message: str = "Feature engineering failed.",
        details: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(
            message=message,
            error_code="ML-002",
            details=details,
        )


class ModelInferenceError(FleetSenseError):

    def __init__(
        self,
        message: str = "Prediction failed.",
        details: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(
            message=message,
            error_code="ML-003",
            details=details,
        )

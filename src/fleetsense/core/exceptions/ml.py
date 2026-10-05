from .base import FleetSenseError


class ModelTrainingError(FleetSenseError):

    def __init__(
        self,
        message: str = "Model training failed.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="ML-001",
        )


class FeatureEngineeringError(FleetSenseError):

    def __init__(
        self,
        message: str = "Feature engineering failed.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="ML-002",
        )


class ModelInferenceError(FleetSenseError):

    def __init__(
        self,
        message: str = "Prediction failed.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="ML-003",
        )
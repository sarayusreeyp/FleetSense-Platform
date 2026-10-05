from .base import FleetSenseError


class DataValidationError(FleetSenseError):

    def __init__(
        self,
        message: str = "Data validation failed.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="DATA-001",
        )


class SchemaMismatchError(FleetSenseError):

    def __init__(
        self,
        message: str = "Incoming schema mismatch.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="DATA-002",
        )


class MissingColumnError(FleetSenseError):

    def __init__(
        self,
        message: str = "Required column missing.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="DATA-003",
        )
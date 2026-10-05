from .base import FleetSenseError


class APIConnectionError(FleetSenseError):
    """Raised when an external API cannot be reached."""

    def __init__(
        self,
        message: str = "Unable to connect to external API.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="API-001",
        )


class APITimeoutError(FleetSenseError):
    """Raised when an API request times out."""

    def __init__(
        self,
        message: str = "API request timed out.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="API-002",
        )


class APIRateLimitError(FleetSenseError):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "API rate limit exceeded.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="API-003",
        )
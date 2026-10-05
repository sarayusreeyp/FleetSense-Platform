"""
Base exception hierarchy for FleetSense.
"""

from __future__ import annotations

from typing import Any


class FleetSenseError(Exception):
    """
    Base exception for all FleetSense-specific errors.

    Attributes
    ----------
    message : str
        Human-readable description.
    error_code : str
        Internal error code.
    details : dict
        Additional metadata.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "FS-000",
        details: dict[str, Any] | None = None,
    ) -> None:

        self.message = message
        self.error_code = error_code
        self.details = details or {}

        super().__init__(message)

    def __str__(self) -> str:
        return (
            f"[{self.error_code}] "
            f"{self.message} "
            f"{self.details}"
        )
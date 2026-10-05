from __future__ import annotations

from typing import Any
from .base import FleetSenseError


class StorageError(FleetSenseError):
    def __init__(
        self,
        message: str = "Storage operation failed.",
        error_code: str = "STORAGE-001",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


class StorageFileNotFoundError(StorageError):
    def __init__(
        self,
        message: str = "Requested file not found in storage.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, error_code="STORAGE-002", details=details)


class StorageConnectionError(StorageError):
    def __init__(
        self,
        message: str = "Failed to connect to storage backend.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, error_code="STORAGE-003", details=details)


class StorageWriteError(StorageError):
    def __init__(
        self,
        message: str = "Failed to write data to storage.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, error_code="STORAGE-004", details=details)
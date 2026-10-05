from .base import FleetSenseError


class StorageError(FleetSenseError):

    def __init__(
        self,
        message: str = "Storage operation failed.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="INFRA-001",
        )


class DatabaseError(FleetSenseError):

    def __init__(
        self,
        message: str = "Database operation failed.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="INFRA-002",
        )


class WorkflowError(FleetSenseError):

    def __init__(
        self,
        message: str = "Workflow execution failed.",
    ) -> None:

        super().__init__(
            message=message,
            error_code="INFRA-003",
        )
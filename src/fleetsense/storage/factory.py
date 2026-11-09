from __future__ import annotations

import os

from fleetsense.core.config import settings
from fleetsense.storage.azure_blob import AzureBlobStorage
from fleetsense.storage.base import Storage
from fleetsense.storage.local import LocalStorage


def get_storage(
    backend: str | None = None,
    root_dir: str | None = None,
    azure_connection_string: str | None = None,
    azure_account_url: str | None = None,
    azure_container_name: str | None = None,
) -> Storage:
    """
    Return the configured storage backend (Local or Azure Blob).

    Parameters
    ----------
    backend : str | None
        Explicit backend name ('local' or 'azure'). If None, uses settings or env var.
    root_dir : str | None
        Local data directory if using local backend.
    azure_connection_string : str | None
        Azure connection string if using Azure backend.
    azure_account_url : str | None
        Azure Blob account URL (for DefaultAzureCredential / Managed Identity).
    azure_container_name : str | None
        Azure container name if using Azure backend.

    Returns
    -------
    Storage
        An instance of LocalStorage or AzureBlobStorage.
    """
    chosen_backend = (
        backend
        or os.getenv("FLEETSENSE_STORAGE_BACKEND")
        or os.getenv("STORAGE_BACKEND")
        or settings.storage.backend
    ).lower()

    # Enforce Azure Blob Storage in production/staging environments
    app_env = (
        os.getenv("FLEETSENSE_APP_ENV")
        or os.getenv("ENVIRONMENT")
        or "development"
    ).lower()
    if app_env in ("production", "staging") and chosen_backend == "local":
        raise ValueError(
            f"Local storage backend is not permitted in '{app_env}' environment. "
            f"Set FLEETSENSE_STORAGE_BACKEND=azure and configure "
            f"AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL for production deployments."
        )

    if chosen_backend == "local":
        target_root = (
            root_dir
            or os.getenv("LOCAL_DATA_DIR")
            or settings.storage.local_data_dir
        )
        return LocalStorage(root_dir=target_root)

    if chosen_backend == "azure":
        conn_str = (
            azure_connection_string
            or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            or settings.storage.azure_connection_string
        )
        acc_url = (
            azure_account_url
            or os.getenv("AZURE_STORAGE_ACCOUNT_URL")
        )
        container = (
            azure_container_name
            or os.getenv("AZURE_STORAGE_CONTAINER_NAME")
            or os.getenv("AZURE_CONTAINER_NAME")
            or settings.storage.azure_container_name
        )
        return AzureBlobStorage(
            connection_string=conn_str,
            account_url=acc_url,
            container_name=container,
        )

    raise ValueError(
        f"Unsupported storage backend: '{chosen_backend}'. "
        f"Supported backends are: 'local', 'azure'."
    )

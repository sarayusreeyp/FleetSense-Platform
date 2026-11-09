import logging
from pathlib import Path

import pytest

# Suppress standard library logging traceback during interpreter finalization when stream closes
logging.raiseExceptions = False


@pytest.fixture(autouse=True)
def test_environment(tmp_path: Path, monkeypatch):
    """
    Ensure unit tests run in an isolated environment:
    - Prefect runs in local ephemeral mode (unsets PREFECT_API_URL so it does not attempt
      to connect to an external server on 127.0.0.1:4200).
    - Sets testing environment flags.
    - Isolates METADATA_DIR so persistent metadata json files don't leak into unit tests.
    """
    test_metadata = tmp_path / "metadata"
    test_metadata.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FLEETSENSE_METADATA_DIR", str(test_metadata))
    monkeypatch.delenv("PREFECT_API_URL", raising=False)
    monkeypatch.setenv("FLEETSENSE_APP_ENV", "test")
    monkeypatch.setenv("PREFECT_LOGGING_TO_API_ENABLED", "false")
    monkeypatch.setenv("PREFECT_LOGGING_TO_API_WHEN_MISSING", "false")
    monkeypatch.setenv("PREFECT_LOGGING_INTERNAL_LEVEL", "WARNING")




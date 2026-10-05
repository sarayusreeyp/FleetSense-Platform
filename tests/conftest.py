import os
import pytest


@pytest.fixture(autouse=True)
def test_environment(monkeypatch):
    """
    Ensure unit tests run in an isolated environment:
    - Prefect runs in local ephemeral mode (unsets PREFECT_API_URL so it does not attempt
      to connect to an external server on 127.0.0.1:4200).
    - Sets testing environment flags.
    """
    monkeypatch.delenv("PREFECT_API_URL", raising=False)
    monkeypatch.setenv("FLEETSENSE_APP_ENV", "testing")

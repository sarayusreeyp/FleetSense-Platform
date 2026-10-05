from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fleetsense.core.config import settings
from fleetsense.core.constants.paths import ARTIFACTS_DIR
from fleetsense.core.logger import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowTracker:
    """
    Manages experiment tracking with MLflow, falling back to local JSON metadata
    when MLflow is unavailable or running in offline environments.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        self.tracking_uri = (
            tracking_uri
            or settings.model.mlflow_tracking_uri
        )
        self.experiment_name = (
            experiment_name
            or settings.model.mlflow_experiment_name
        )
        self.active_run = None
        self.run_metadata: dict[str, Any] = {
            "params": {},
            "metrics": {},
            "artifacts": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self.enabled = True
            except Exception as exc:
                logger.warning(
                    "MLflow setup failed (%s). Falling back to local artifact logging.",
                    exc,
                )
                self.enabled = False
        else:
            self.enabled = False
            logger.info("MLflow not installed. Using local JSON metadata tracker.")

    def start_run(self, run_name: str | None = None) -> Any:
        self.run_metadata["run_name"] = run_name or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if self.enabled:
            try:
                self.active_run = mlflow.start_run(run_name=self.run_metadata["run_name"])
                return self.active_run
            except Exception as exc:
                logger.warning("Failed to start MLflow run: %s", exc)
                self.enabled = False
        return self

    def log_params(self, params: dict[str, Any]) -> None:
        self.run_metadata["params"].update(params)
        if self.enabled:
            try:
                mlflow.log_params(params)
            except Exception as exc:
                logger.warning("MLflow log_params failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.run_metadata["metrics"].update(metrics)
        if self.enabled:
            try:
                mlflow.log_metrics(metrics)
            except Exception as exc:
                logger.warning("MLflow log_metrics failed: %s", exc)

    def log_artifact(self, local_path: Path | str) -> None:
        path_str = str(local_path)
        self.run_metadata["artifacts"].append(path_str)
        if self.enabled:
            try:
                mlflow.log_artifact(path_str)
            except Exception as exc:
                logger.warning("MLflow log_artifact failed: %s", exc)

    def end_run(self) -> dict[str, Any]:
        self.run_metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
        if self.enabled and self.active_run:
            try:
                mlflow.end_run()
            except Exception as exc:
                logger.warning("MLflow end_run failed: %s", exc)

        # Always persist run metadata to local artifacts directory
        meta_dir = ARTIFACTS_DIR / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        run_file = meta_dir / f"{self.run_metadata['run_name']}.json"

        with run_file.open("w", encoding="utf-8") as f:
            json.dump(self.run_metadata, f, indent=2)

        logger.info("Logged run metadata to %s", run_file)
        return self.run_metadata

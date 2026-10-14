from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fleetsense.core.config import settings
from fleetsense.core.constants.paths import ARTIFACTS_DIR, PROJECT_ROOT
from fleetsense.core.logger import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _resolve_tracking_uri() -> str:
    """
    Resolve the MLflow tracking URI from environment or config.

    Priority:
      1. MLFLOW_TRACKING_URI env var  (e.g., http://mlflow:5000 or postgresql://)
      2. settings.model.mlflow_tracking_uri from model.yaml
      3. Fallback to local SQLite (development only)
    """
    uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or settings.model.mlflow_tracking_uri
        or "sqlite:///metadata/mlflow.db"
    )

    # Resolve relative SQLite URIs against PROJECT_ROOT to ensure single source of truth
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        rel_part = uri[len("sqlite:///"):]
        if not (len(rel_part) > 2 and rel_part[1] == ":") and not rel_part.startswith("/"):
            abs_db = (PROJECT_ROOT / rel_part).resolve()
            abs_db.parent.mkdir(parents=True, exist_ok=True)
            uri = f"sqlite:///{abs_db.as_posix()}"

    env = settings.app.environment
    if uri.startswith("sqlite://") and env in ("production", "staging"):
        logger.warning(
            "MLflow is using a LOCAL SQLite database (%s) in '%s' environment. "
            "Set MLFLOW_TRACKING_URI to your remote MLflow server.",
            uri,
            env,
        )

    return uri


class MLflowTracker:
    """
    Manages experiment tracking and Model Registry integration with MLflow.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or _resolve_tracking_uri()
        self.experiment_name = (
            experiment_name
            or os.getenv("MLFLOW_EXPERIMENT_NAME")
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
                logger.info(
                    "MLflow tracking enabled — URI=%s experiment=%s",
                    self.tracking_uri,
                    self.experiment_name,
                )
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

    def log_and_register_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
        alias: str = "champion",
        challenger_metric: float | None = None,
        metric_name: str = "val_mae",
        higher_is_better: bool = False,
        tolerance: float = 0.05,
    ) -> Any:
        """
        Log scikit-learn model to MLflow and register it in the MLflow Model Registry.
        Performs Champion vs Challenger comparison when challenger_metric is provided.
        """
        reg_name = registered_model_name or settings.model.name
        if self.enabled and self.active_run:
            try:
                logger.info(
                    "Logging model to MLflow and registering under name '%s'...",
                    reg_name,
                )
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=reg_name,
                    signature=signature,
                    input_example=input_example,
                )
                self.run_metadata["registered_model_name"] = reg_name
                self.run_metadata["model_uri"] = model_info.model_uri

                # Champion vs Challenger model alias resolution
                if alias:
                    try:
                        from mlflow.tracking import MlflowClient
                        client = MlflowClient(tracking_uri=self.tracking_uri)
                        version_to_alias = None
                        if hasattr(model_info, "registered_model_version") and model_info.registered_model_version:
                            version_to_alias = str(model_info.registered_model_version)
                        else:
                            latest_versions = client.get_latest_versions(reg_name)
                            if latest_versions:
                                version_to_alias = str(latest_versions[0].version)

                        if version_to_alias:
                            target_alias = alias
                            if challenger_metric is not None and alias == "champion":
                                try:
                                    current_champion = client.get_model_version_by_alias(reg_name, "champion")
                                    champ_run = client.get_run(current_champion.run_id)
                                    champ_metric = champ_run.data.metrics.get(metric_name)

                                    if champ_metric is not None:
                                        if higher_is_better:
                                            beats_champ = challenger_metric >= champ_metric * (1.0 - tolerance)
                                        else:
                                            beats_champ = challenger_metric <= champ_metric * (1.0 + tolerance)

                                        if beats_champ:
                                            logger.info(
                                                "Challenger (v%s, %s=%.4f) beat or matched Champion (v%s, %s=%.4f). Promoting to @champion.",
                                                version_to_alias, metric_name, challenger_metric, current_champion.version, metric_name, champ_metric
                                            )
                                            target_alias = "champion"
                                        else:
                                            logger.warning(
                                                "Challenger (v%s, %s=%.4f) did not beat Champion (v%s, %s=%.4f). Tagged as @challenger.",
                                                version_to_alias, metric_name, challenger_metric, current_champion.version, metric_name, champ_metric
                                            )
                                            target_alias = "challenger"
                                except Exception as comp_err:
                                    logger.info("No prior champion found or comparison error (%s); promoting to @champion.", comp_err)
                                    target_alias = "champion"

                            client.set_registered_model_alias(
                                name=reg_name,
                                alias=target_alias,
                                version=version_to_alias,
                            )
                            logger.info("Assigned alias '@%s' to version %s of '%s'", target_alias, version_to_alias, reg_name)
                    except Exception as alias_exc:
                        logger.warning("Could not set model registry alias: %s", alias_exc)

                logger.info("Successfully registered model in MLflow: %s", model_info.model_uri)
                return model_info
            except Exception as exc:
                logger.warning("MLflow log_and_register_model failed: %s", exc)
        return None

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

from pathlib import Path
from fleetsense.ml.tracking import MLflowTracker


def test_mlflow_tracker_run(tmp_path: Path):
    tracker = MLflowTracker()
    tracker.start_run(run_name="test_run_unit")

    tracker.log_params({"n_estimators": 50, "algorithm": "HistGradientBoosting"})
    tracker.log_metrics({"mae": 3.14, "rmse": 4.56, "r2": 0.85})

    artifact_file = tmp_path / "dummy_artifact.txt"
    artifact_file.write_text("dummy model weights")
    tracker.log_artifact(artifact_file)

    run_metadata = tracker.end_run()

    assert run_metadata["run_name"] == "test_run_unit"
    assert run_metadata["params"]["algorithm"] == "HistGradientBoosting"
    assert run_metadata["metrics"]["mae"] == 3.14
    assert len(run_metadata["artifacts"]) == 1
    assert "started_at" in run_metadata
    assert "ended_at" in run_metadata

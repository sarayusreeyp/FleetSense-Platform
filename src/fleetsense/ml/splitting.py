from typing import Final

import pandas as pd

from fleetsense.core.exceptions.data import DataValidationError


DEFAULT_TRAIN_RATIO: Final[float] = 0.70
DEFAULT_VALIDATION_RATIO: Final[float] = 0.15
DEFAULT_TEST_RATIO: Final[float] = 0.15


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Split features and target chronologically.

    The input must already be ordered chronologically.
    The function does not shuffle the data.
    """

    _validate_inputs(
        features=features,
        target=target,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    row_count = len(features)

    train_end = int(row_count * train_ratio)

    validation_end = train_end + int(
        row_count * validation_ratio
    )

    X_train = features.iloc[:train_end].copy()
    y_train = target.iloc[:train_end].copy()

    X_validation = features.iloc[
        train_end:validation_end
    ].copy()

    y_validation = target.iloc[
        train_end:validation_end
    ].copy()

    X_test = features.iloc[
        validation_end:
    ].copy()

    y_test = target.iloc[
        validation_end:
    ].copy()

    _validate_split_sizes(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        expected_total=row_count,
    )

    return {
        "train": (X_train, y_train),
        "validation": (
            X_validation,
            y_validation,
        ),
        "test": (X_test, y_test),
    }


def _validate_inputs(
    features: pd.DataFrame,
    target: pd.Series,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    """Validate splitting inputs."""

    if not isinstance(features, pd.DataFrame):
        raise DataValidationError(
            message="Features must be a pandas DataFrame."
        )

    if not isinstance(target, pd.Series):
        raise DataValidationError(
            message="Target must be a pandas Series."
        )

    if features.empty:
        raise DataValidationError(
            message="Cannot split an empty feature dataframe."
        )

    if target.empty:
        raise DataValidationError(
            message="Cannot split an empty target series."
        )

    if len(features) != len(target):
        raise DataValidationError(
            message=(
                "Features and target must contain "
                "the same number of rows."
            )
        )

    ratios = (
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    if any(ratio <= 0 for ratio in ratios):
        raise DataValidationError(
            message="All split ratios must be greater than zero."
        )

    if not abs(sum(ratios) - 1.0) < 1e-9:
        raise DataValidationError(
            message="Split ratios must sum to 1.0."
        )


def _validate_split_sizes(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    expected_total: int,
) -> None:
    """Ensure no rows were lost during splitting."""

    actual_total = (
        len(X_train)
        + len(X_validation)
        + len(X_test)
    )

    if actual_total != expected_total:
        raise DataValidationError(
            message=(
                "Dataset splitting lost rows: "
                f"expected {expected_total}, "
                f"got {actual_total}."
            )
        )

    if (
        X_train.empty
        or X_validation.empty
        or X_test.empty
    ):
        raise DataValidationError(
            message=(
                "Train, validation, and test "
                "sets must all contain rows."
            )
        )
from pathlib import Path
from time import sleep

import requests

from fleetsense.core.exceptions.data import DataValidationError


class TLCDownloader:
    """Downloads NYC TLC monthly trip data."""

    BASE_URL = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data"
    )

    def __init__(
        self,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.max_retries = max_retries

    def build_url(
        self,
        dataset: str,
        year: int,
        month: int,
    ) -> str:
        """Build the TLC monthly Parquet URL."""

        if not 1 <= month <= 12:
            raise ValueError(
                "Month must be between 1 and 12."
            )

        filename = (
            f"{dataset}_tripdata_"
            f"{year:04d}-{month:02d}.parquet"
        )

        return f"{self.BASE_URL}/{filename}"

    def is_available(
        self,
        dataset: str,
        year: int,
        month: int,
    ) -> bool:
        """Check whether a TLC monthly file is available."""

        url = self.build_url(
            dataset=dataset,
            year=year,
            month=month,
        )

        try:
            response = requests.head(
                url,
                timeout=self.timeout,
                allow_redirects=True,
            )
            return response.status_code == 200

        except requests.RequestException:
            return False

    def download(
        self,
        dataset: str,
        year: int,
        month: int,
        output_path: Path,
    ) -> Path:
        """Download one monthly TLC file."""

        url = self.build_url(
            dataset=dataset,
            year=year,
            month=month,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            try:
                response = requests.get(
                    url,
                    timeout=self.timeout,
                    stream=True,
                )

                response.raise_for_status()

                with temporary_path.open(
                    "wb"
                ) as file:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            file.write(chunk)

                if temporary_path.stat().st_size == 0:
                    raise DataValidationError(
                        message=(
                            f"Downloaded file is empty: {url}"
                        )
                    )

                temporary_path.replace(
                    output_path
                )

                return output_path

            except (
                requests.RequestException,
                DataValidationError,
            ) as exc:

                if temporary_path.exists():
                    temporary_path.unlink()

                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Failed to download {url} "
                        f"after {self.max_retries} attempts."
                    ) from exc

                sleep(2 ** (attempt - 1))

        raise RuntimeError(
            f"Failed to download {url}."
        )
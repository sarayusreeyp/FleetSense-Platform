from pathlib import Path

import pytest

from fleetsense.etl.downloader import TLCDownloader


def test_build_url():
    downloader = TLCDownloader()

    url = downloader.build_url(
        dataset="yellow",
        year=2026,
        month=1,
    )

    assert (
        url
        == "https://d37ci6vzurychx.cloudfront.net/"
           "trip-data/yellow_tripdata_2026-01.parquet"
    )


def test_build_url_rejects_invalid_month():
    downloader = TLCDownloader()

    with pytest.raises(ValueError):
        downloader.build_url(
            dataset="yellow",
            year=2026,
            month=13,
        )


def test_download_success(tmp_path: Path, monkeypatch):
    downloader = TLCDownloader()

    class FakeResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"fake parquet data"

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "fleetsense.etl.downloader.requests.get",
        fake_get,
    )

    output_path = (
        tmp_path
        / "yellow_tripdata_2026-01.parquet"
    )

    result = downloader.download(
        dataset="yellow",
        year=2026,
        month=1,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.read_bytes() == b"fake parquet data"


def test_download_rejects_empty_file(tmp_path: Path, monkeypatch):
    downloader = TLCDownloader(
        max_retries=1,
    )

    class FakeResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            return
            yield

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "fleetsense.etl.downloader.requests.get",
        fake_get,
    )

    output_path = (
        tmp_path
        / "yellow_tripdata_2026-01.parquet"
    )

    with pytest.raises(RuntimeError):
        downloader.download(
            dataset="yellow",
            year=2026,
            month=1,
            output_path=output_path,
        )

    assert not output_path.exists()
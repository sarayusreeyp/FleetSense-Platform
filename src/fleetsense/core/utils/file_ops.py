from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> Path:
    ensure_parent_dir(output_path)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(dict(record), ensure_ascii=False) + "\n")

    return output_path
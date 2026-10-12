from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if "models" not in manifest or not isinstance(manifest["models"], list):
        raise ValueError(f"Manifest {path} must contain a models list.")
    return manifest


def run_file_name(label: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in label)
    return f"{safe}.jsonl"


def safe_name(label: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in label)

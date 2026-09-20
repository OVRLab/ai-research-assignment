"""Shared paths and provenance for the Granite exercise."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def baseline() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            CONFIG["model"],
            revision=CONFIG["revision"],
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "LICENSE*", "README.md"],
        )
    )


def provenance() -> dict:
    return {
        "base_model": CONFIG["model"],
        "base_revision": CONFIG["revision"],
        "config_sha256": digest(ROOT / "config.json"),
    }

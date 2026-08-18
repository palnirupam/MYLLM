"""Content-addressed artifact and runtime identity helpers for V2."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def source_revision() -> str:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], stderr=subprocess.DEVNULL)
        dirty = hashlib.sha256(diff).hexdigest()[:16] if diff else "clean"
        return f"{head}+{dirty}"
    except Exception:
        return "unknown"


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _module_version("torch"),
        "tokenizers": _module_version("tokenizers"),
        "cuda": _cuda_metadata(),
    }


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def _cuda_metadata() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "version": torch.version.cuda,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception:
        return {"available": False}


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def atomic_replace_dir(temp_dir: Path, final_dir: Path) -> None:
    """Publish a complete directory in one rename operation."""
    if final_dir.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {final_dir}")
    temp_dir.rename(final_dir)

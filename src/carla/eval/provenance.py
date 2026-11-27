"""Capture runnable source, not just hashes of an uncommitted working tree."""

import hashlib
import zipfile
from pathlib import Path


def archive_sources(output):
    root = Path(__file__).resolve().parents[3]
    destination = Path(output) / "source.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in ("src", "adapters", "scripts", "configs"):
            for path in sorted((root / directory).rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix not in (".pyc", ".class", ".jar")
                ):
                    archive.write(path, path.relative_to(root))
        for name in ("pyproject.toml", "uv.lock"):
            archive.write(root / name, name)
    return {
        "source_archive": destination.name,
        "source_archive_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }

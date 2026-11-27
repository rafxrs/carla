"""Install pinned optional Catanatron and build the original JSettlers sidecar.

Run with the same Python interpreter used for CARLA. Downloads are cached under
outputs, never imported from an unpinned default branch.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

CATANATRON = "ecf931181b9a65bb4116a2153fb78c16f1438e00"
JAR_SHA256 = "88d31577d12242cccf5b44ba378ec70a8ac51fac1359d89ed471c1e0068c2844"
JAR_URL = "https://github.com/jdmonin/JSettlers2/releases/download/release-2.6.10/jsettlers-2.6.10-full.jar"


def setup(destination, *, catanatron=True, java=True):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "catanatron_commit": CATANATRON,
        "jsettlers_version": "2.6.10",
        "jsettlers_sha256": JAR_SHA256,
    }
    if catanatron:
        # uv is optional: install into precisely the invoking interpreter.
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                f"catanatron @ git+https://github.com/bcollazo/catanatron.git@{CATANATRON}",
            ],
            check=True,
        )
    if java:
        jar = destination / "jsettlers-2.6.10-full.jar"
        if not jar.exists():
            urllib.request.urlretrieve(JAR_URL, jar)
        if hashlib.sha256(jar.read_bytes()).hexdigest() != JAR_SHA256:
            raise ValueError("JSettlers JAR hash mismatch; refusing to execute")
        classes = destination / "jsettlers-classes"
        classes.mkdir(exist_ok=True)
        sources = list(
            (Path(__file__).resolve().parents[1] / "adapters/jsettlers/src").rglob("*.java")
        )
        subprocess.run(
            [
                "javac",
                "-encoding",
                "UTF-8",
                "-cp",
                str(jar),
                "-d",
                str(classes),
                *[str(p) for p in sources],
            ],
            check=True,
        )
        manifest.update(jar=str(jar), classes=str(classes))
        print(f'$env:JSETTLERS_JAR = "{jar}"')
        print(f'$env:CARLA_JSETTLERS_CLASSES = "{classes}"')
        print(f'export JSETTLERS_JAR="{jar}" CARLA_JSETTLERS_CLASSES="{classes}"')
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="outputs/benchmark-deps")
    parser.add_argument("--skip-catanatron", action="store_true")
    parser.add_argument("--skip-java", action="store_true")
    args = parser.parse_args()
    setup(args.destination, catanatron=not args.skip_catanatron, java=not args.skip_java)

"""Original JSettlers strategies in a Java sidecar with CARLA phase adaptation."""

import os
from pathlib import Path

from carla.agents.catanatron.coordinates import corner_keys
from carla.agents.external import ExternalAgent
from carla.engine.topology import EDGE_CORNERS, PORT_CORNERS, TILE_AXIAL


class JSettlersAgent(ExternalAgent):
    def __init__(self, strategy="smart", *, jar=None, classes=None, timeout=30):
        if strategy not in ("smart", "fast"):
            raise ValueError("JSettlers strategy must be smart or fast")
        jar = jar or os.environ.get("JSETTLERS_JAR")
        classes = classes or os.environ.get("CARLA_JSETTLERS_CLASSES")
        if not jar or not classes or not Path(jar).is_file() or not Path(classes).is_dir():
            raise RuntimeError(
                "run scripts/setup_benchmarks.py and set JSETTLERS_JAR and CARLA_JSETTLERS_CLASSES"
            )
        super().__init__(
            [
                "java",
                # HotSpot prints VM warnings to stdout by default, which is the reply channel.
                "-XX:+DisplayVMOutputToStderr",
                "-cp",
                os.pathsep.join((str(classes), str(jar))),
                "osp.jsettlers.CarlaAgent",
                strategy,
            ],
            timeout,
        )
        self.name = f"jsettlers-{strategy}-adapted"

    def _request(self, kind, **payload):
        if kind == "before":
            keys = corner_keys()
            payload["topology"] = {
                "corners": keys,
                "tiles": TILE_AXIAL,
                "edges": [sorted((keys[a], keys[b])) for a, b in EDGE_CORNERS],
                "ports": payload["view"]["state"]["board"].get("port_corners", PORT_CORNERS),
            }
        return super()._request(kind, **payload)

"""Minimal CARLA protocol v1 bot; stdout is exclusively protocol traffic."""

import json
import random
import sys

rng = random.Random()
for line in sys.stdin:
    message = json.loads(line)
    kind = message["type"]
    if kind == "step":
        continue
    reply = {"type": kind, "request": message["request"]}
    if kind == "handshake":
        reply.update(protocol="carla-agent", version=1, capabilities={"players": [4]})
    elif kind == "before":
        rng.seed(message["seed"])
    elif kind == "decide":
        reply["action"] = rng.choice(message["view"]["legal"])
    print(json.dumps(reply), flush=True)

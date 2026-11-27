"""Run supported CARLA agents under the original JSettlers referee via OSP."""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

from carla.eval.provenance import archive_sources


def terminate_process_tree(process):
    """Stop the JVM AND its agent children, including Windows launcher shims."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def run(games, output, trading=False, timeout=300, agent="heuristic"):
    if games < 1 or timeout <= 0:
        raise ValueError("games and timeout must be positive")
    from carla.agents.osp import SUPPORTED

    if agent not in SUPPORTED and not agent.startswith("ppo:"):
        raise ValueError("unsupported external-referee agent")
    if (agent.endswith("-native") or agent.startswith("ppo:")) and trading:
        raise ValueError("event-history external-referee profile requires domestic trading off")
    jar = os.environ["JSETTLERS_JAR"]
    classes = os.environ["CARLA_JSETTLERS_CLASSES"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "agent-logs"
    logs.mkdir(exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # Quoted: checkpoint paths may contain spaces (AgentProcess.split honours quotes).
    command = f'"{Path(sys.executable).as_posix()}" -m carla.agents.osp --agent "{agent}"'
    properties = {
        "jsettlers.startrobots": "4",
        "jsettlers.bots.start3p": "1,osp.jsettlers.OSPRobotClient",
        "jsettlers.bots.percent3p": "100",
        "jsettlers.bots.botgames.total": str(games),
        "jsettlers.bots.botgames.parallel": "1",
        "jsettlers.bots.botgames.shutdown": "Y",
        "jsettlers.bots.botgames.wait_sec": "1",
        "jsettlers.bots.fast_pause_percent": "1",
        "jsettlers.bots.timeout.turn": "30",
        "jsettlers.allow.debug": "N",
        "jsettlers.port": str(port),
    }
    if not trading:
        properties["jsettlers.gameopt.NT"] = "t"
    arguments = [
        "java",
        f"-Dosp.agent={command}",
        f"-Dosp.log={logs}",
        f"-Dosp.record={output / 'records'}",
        "-Dosp.bots=smart",
        "-Dosp.max_failed_games=1",
        "-cp",
        os.pathsep.join((classes, jar)),
        "osp.jsettlers.OSPServer",
        *[f"-D{k}={v}" for k, v in properties.items()],
    ]
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "referee": "JSettlers 2.6.10",
                "agent": agent,
                "trading": trading,
                "command": arguments,
                "jar_sha256": hashlib.sha256(Path(jar).read_bytes()).hexdigest(),
                **archive_sources(output),
            },
            indent=2,
        )
        + "\n"
    )
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            arguments,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process_tree(process)
            code = -1
        except BaseException:
            terminate_process_tree(process)
            raise
    text = (output / "server.log").read_text(encoding="utf-8", errors="replace")
    results = []
    for line in text.splitlines():
        if '"osp-result"' in line:
            results.append(json.loads(line[line.index("{") :]))
    summary = {
        "exit_code": code,
        "timed_out": code == -1,
        "scheduled": games,
        "completed_records": len(results),
        "results": results,
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--output", default="outputs/benchmarks/jsettlers-referee")
    parser.add_argument("--trading", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--agent",
        default="heuristic",
        help="heuristic, random, random_flat, jsettlers-{fast,smart}-native or ppo:<checkpoint>",
    )
    args = parser.parse_args()
    run(args.games, args.output, args.trading, args.timeout, args.agent)

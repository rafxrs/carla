"""Strict JSON-lines agent client. No replacement policy on errors or timeouts."""

import contextlib
import json
import queue
import subprocess
import threading
from collections.abc import Sequence

from carla.agents.api import PROTOCOL, VERSION, AgentCapabilities, AgentView, event_json
from carla.agents.base import Agent
from carla.engine.actions import Action
from carla.engine.game import Game


class ExternalAgent(Agent):
    def __init__(self, command: Sequence[str], timeout: float = 10.0):
        if isinstance(command, (str, bytes)) or not command or timeout <= 0:
            raise ValueError("external agents need a command and positive timeout")
        self.command = list(command)
        self.timeout = timeout
        self.name = "external"
        self.process = None
        self._sequence = 0
        self._replies = queue.Queue(maxsize=4)
        self._pending_write = None

    def _read(self, process, replies):
        try:
            while line := process.stdout.readline(2_000_001):
                if len(line) > 2_000_000:
                    raise ValueError("external reply exceeds size limit")
                try:
                    replies.put_nowait(json.loads(line))
                except json.JSONDecodeError:
                    # Name the line: a runtime's own output on stdout (e.g. a JVM warning)
                    # is otherwise indistinguishable from a malformed reply.
                    raise ValueError(
                        f"external agent wrote non-JSON output: {line[:200]!r}"
                    ) from None
        except Exception as error:
            with contextlib.suppress(queue.Full):
                replies.put_nowait(error)
        finally:
            with contextlib.suppress(queue.Full):
                replies.put_nowait(EOFError("external agent exited"))

    def _send(self, kind, **payload):
        assert self.process is not None and self.process.stdin is not None
        done = threading.Event()
        self._pending_write = done
        errors = []
        self._writes.put(
            (json.dumps({"type": kind, **payload}) + "\n", done, errors), timeout=self.timeout
        )
        if not done.wait(self.timeout):
            # Killing the peer releases an OS pipe write blocked on a non-reading bot.
            self.process.kill()
            raise TimeoutError(f"external agent stopped reading on {kind}")
        if errors:
            raise errors[0]

    @staticmethod
    def _write(process, writes):
        while (item := writes.get()) is not None:
            text, done, errors = item
            try:
                process.stdin.write(text)
                process.stdin.flush()
            except Exception as error:
                errors.append(error)
            finally:
                done.set()

    def _request(self, kind, **payload):
        self._sequence += 1
        self._send(kind, request=self._sequence, **payload)
        try:
            reply = self._replies.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError(f"external agent timed out on {kind}") from None
        if isinstance(reply, Exception):
            raise reply
        if reply.get("request") != self._sequence or reply.get("type") != kind:
            raise ValueError(f"out-of-sequence external reply: {reply}")
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def reset(self, game: Game, seat: int, seed: int):
        self.close()
        self._sequence = 0
        self._replies = queue.Queue(maxsize=4)
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._thread = threading.Thread(
            target=self._read, args=(self.process, self._replies), daemon=True
        )
        self._thread.start()
        self._writes = queue.Queue(maxsize=1)
        self._writer = threading.Thread(
            target=self._write, args=(self.process, self._writes), daemon=True
        )
        self._writer.start()
        reply = self._request("handshake", protocol=PROTOCOL, version=VERSION)
        if reply.get("protocol") != PROTOCOL or reply.get("version") != VERSION:
            raise ValueError("unsupported external protocol")
        AgentCapabilities(**reply.get("capabilities", {})).validate(game)
        self._request("before", seed=seed, view=AgentView.from_game(game, seat).to_json())

    def observe(self, events, step):
        self._send("step", step=step, events=[event_json(e) for e in events])

    def act(self, game, seat, legal):
        reply = self._request("decide", view=AgentView.from_game(game, seat).to_json())
        action = Action.from_json(reply["action"])
        if not game.is_legal(action):
            raise ValueError(f"external action is illegal: {action}")
        return action

    def finish(self, result):
        self._request("after", result=result)

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        if (
            self._pending_write is not None
            and not self._pending_write.is_set()
            and process.poll() is None
        ):
            # Cancellation may arrive while a writer owns the stdin lock.
            process.kill()
        if process.stdin:
            with contextlib.suppress(BrokenPipeError, OSError):
                process.stdin.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stdout:
            process.stdout.close()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1)
        if hasattr(self, "_writer"):
            self._writes.put(None, timeout=1)
            self._writer.join(timeout=1)

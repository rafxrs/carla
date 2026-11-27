"""Dedicated, stateful Java oracle for native-port differential checks.

This process is separate from scored opponents: diagnostic operations such as
planning and opening selection mutate the original robot's internal state.
"""

from carla.agents.api import AgentView
from carla.agents.jsettlers import JSettlersAgent


class JSettlersReference:
    def __init__(self, strategy="fast", **runtime):
        self._agent = JSettlersAgent(strategy, **runtime)

    def reset(self, game, seat, seed):
        self.seat = seat
        self._agent.reset(game, seat, seed)

    def inspect(self, game, operation="snapshot", **parameters):
        reply = self._agent._request(
            "inspect",
            view=AgentView.from_game(game, self.seat).to_json(),
            operation=operation,
            **parameters,
        )
        return reply["diagnostic"]

    def observe(self, events, step):
        self._agent.observe(events, step)

    def close(self):
        self._agent.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

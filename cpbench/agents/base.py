from __future__ import annotations

import random

from ..board import Board


class Agent:
    """Agents map an observation dict (see observation.observe) to an action dict."""

    def __init__(self, agent_id: str, board: Board, seed: int | None = None):
        self.id = agent_id
        self.board = board
        self.rng = random.Random(seed)

    def act(self, obs: dict) -> dict:
        raise NotImplementedError


class RandomAgent(Agent):
    """Uniformly random legal actions; never talks."""

    def act(self, obs: dict) -> dict:
        req = obs["request"]
        k = req["kind"]
        if k == "place":
            return {"station": self.rng.choice(req["legal"])}
        if k == "thief_move":
            return {"path": self.rng.choice(req["legal"])}
        if k == "police_action":
            return {"type": self.rng.choice(req["legal_types"]),
                    "station": self.rng.choice(req["legal_stations"])}
        if k == "report":
            return {"team_claim": self.rng.choice(req["legal_claims"]),
                    "share_public": self.rng.random() < 0.5}
        if k == "vote":
            return {"target": self.rng.choice(req["legal"])}
        return {}

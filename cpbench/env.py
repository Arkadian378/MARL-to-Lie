"""Multi-agent environment API.

A simultaneous-move ("parallel") interface where the set of acting agents
changes by phase::

    env = CorruptPoliceEnv(GameConfig(seed=0))
    obs = env.reset()
    while not env.done:
        actions = {aid: agents[aid].act(o) for aid, o in obs.items()}
        obs, rewards, done, info = env.step(actions)

``obs`` only contains agents that must act now. ``rewards`` are zero until
the end, then equal each agent's final piece payout (the show's currency).
"""
from __future__ import annotations

from dataclasses import replace

from .config import GameConfig
from .engine import Game
from .observation import observe, render_text, render_request
from . import transcript as _transcript


class CorruptPoliceEnv:
    metadata = {"name": "corrupt_police_v0"}

    def __init__(self, config: GameConfig | None = None, names: list[str] | None = None,
                 text_obs: bool = False):
        self.config = config or GameConfig()
        self.names = names
        self.text_obs = text_obs
        self.game: Game | None = None

    # ----------------------------------------------------------------- api
    @property
    def possible_agents(self) -> list[str]:
        return list(self.game.players) if self.game else []

    @property
    def done(self) -> bool:
        return bool(self.game and self.game.done)

    def reset(self, seed: int | None = None) -> dict[str, dict]:
        cfg = replace(self.config, seed=seed) if seed is not None else self.config
        self.game = Game(cfg, names=self.names)
        return self._obs()

    def observe(self, agent: str) -> dict:
        o = observe(self.game, agent)
        if self.text_obs:
            o["text"] = render_text(o)
            if o["request"]:
                o["text"] += "\n\n" + render_request(o["request"])
        return o

    def step(self, actions: dict[str, dict]):
        self.game.step(actions)
        rewards = {a: 0.0 for a in self.game.players}
        if self.game.done:
            rewards = {a: float(v) for a, v in self.game.payout.items()}
        info = {"phase": self.game.phase, "round": self.game.round}
        if self.game.done:
            info["summary"] = self.game.summary
        return self._obs(), rewards, self.game.done, info

    def _obs(self) -> dict[str, dict]:
        return {a: self.observe(a) for a in self.game.pending}

    def transcript(self, **meta) -> dict:
        return _transcript.build(self.game, **meta)

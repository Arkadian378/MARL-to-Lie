from __future__ import annotations

import zlib
from typing import Callable

from .config import GameConfig
from .env import CorruptPoliceEnv


def run_episode(config: GameConfig | None = None, agent_factory: Callable | None = None,
                seed: int | None = None, **meta) -> tuple[CorruptPoliceEnv, dict]:
    """Play one full game. ``agent_factory(agent_id, board, seed) -> Agent``."""
    from .agents import HeuristicAgent
    env = CorruptPoliceEnv(config)
    obs = env.reset(seed=seed)
    factory = agent_factory or (lambda aid, board, s: HeuristicAgent(aid, board, s))
    base = seed or 0
    agents = {aid: factory(aid, env.game.board, zlib.crc32(f"{base}:{aid}".encode()))
              for aid in env.possible_agents}
    while not env.done:
        actions = {aid: agents[aid].act(o) for aid, o in obs.items()}
        obs, rewards, done, info = env.step(actions)
    return env, env.transcript(seed=seed, **meta)

"""Batched multi-game rollouts for RL.

Many games are stepped together. At every tick the seats played by an adapter (a trained
role, or a frozen snapshot) are collected across all games and generated in one batch; all
other seats (heuristic, random, API models) act immediately. Every decision of a *learning*
seat becomes a ``Sample`` that the trainer scores with the final game payout.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from cpbench import CorruptPoliceEnv  # noqa: E402
from cpbench.agents import RandomAgent  # noqa: E402
from cpbench.agents.llm import SYSTEM, build_prompt, parse_action  # noqa: E402
from common import game_config, make_agent, agent_seed  # noqa: E402


@dataclass
class Sample:
    adapter: str
    game: int
    agent: str
    kind: str
    prompt_ids: list[int]
    completion_ids: list[int]
    parse_ok: bool
    advantage: float = 0.0


@dataclass
class SeatPlan:
    """How one role is played in one game."""
    adapter: str | None = None     # adapter name -> policy seat; None -> ``spec`` agent
    learn: bool = False            # record samples for training
    spec: dict = field(default_factory=lambda: {"type": "heuristic"})


@dataclass
class GameResult:
    game: int
    seed: int
    env: CorruptPoliceEnv
    learners: list[str]            # agent ids whose samples are trained on
    parse_failures: dict


def run_rollouts(policy, generator, game_spec: dict, plans: list[dict[str, SeatPlan]], seeds: list[int],
                 history_events: int = 80, backends: dict | None = None, max_prompt_tokens: int = 7000,
                 system: str = SYSTEM, train_kinds: set | None = None):
    """``plans[i]`` maps role -> SeatPlan for game i (same length as ``seeds``)."""
    envs, obs, agents, fallback, seat_plan = [], [], [], [], []
    for gi, (seed, plan) in enumerate(zip(seeds, plans)):
        env = CorruptPoliceEnv(game_config(game_spec, seed))
        o = env.reset()
        g = env.game
        ag, fb, sp = {}, {}, {}
        for aid, p in g.players.items():
            sp[aid] = plan.get(p.role, SeatPlan())
            fb[aid] = RandomAgent(aid, g.board, agent_seed(seed, aid) + 1)
            if sp[aid].adapter is None:
                ag[aid] = make_agent(sp[aid].spec, aid, g.board, seed, backends or {}, history_events)
        envs.append(env), obs.append(o), agents.append(ag), fallback.append(fb), seat_plan.append(sp)

    samples: list[Sample] = []
    pfail = [dict() for _ in envs]
    too_long = 0
    while True:
        active = [i for i, e in enumerate(envs) if not e.done]
        if not active:
            break
        actions = {i: {} for i in active}
        reqs = []                                   # (game, agent, adapter, prompt_text)
        for i in active:
            for aid, o in obs[i].items():
                plan = seat_plan[i][aid]
                if plan.adapter is None:
                    actions[i][aid] = agents[i][aid].act(o)
                else:
                    reqs.append((i, aid, plan.adapter, policy.chat_prompt(system, build_prompt(o, history_events))))
        if reqs:
            outs = generator.generate([r[3] for r in reqs], [r[2] for r in reqs])
            for (i, aid, adapter, prompt), (text, comp_ids) in zip(reqs, outs):
                o = obs[i][aid]
                a, _ = parse_action(text)
                ok = a is not None
                if not ok:
                    pfail[i][aid] = pfail[i].get(aid, 0) + 1
                    a = fallback[i][aid].act(o)
                elif "thought" in a:
                    a["_thought"] = a.pop("thought")
                actions[i][aid] = a
                plan = seat_plan[i][aid]
                kind = o["request"]["kind"]
                if plan.learn and comp_ids and (train_kinds is None or kind in train_kinds):
                    p_ids = policy.encode(prompt)
                    if len(p_ids) > max_prompt_tokens:
                        too_long += 1
                        continue
                    samples.append(Sample(adapter, i, aid, kind, p_ids, list(comp_ids), ok))
        for i in active:
            obs[i], _, _, _ = envs[i].step(actions[i])

    results = []
    for i, env in enumerate(envs):
        learners = [aid for aid, p in seat_plan[i].items() if p.learn]
        results.append(GameResult(i, seeds[i], env, learners, pfail[i]))
    if too_long:
        print(f"[rollout] {too_long} decisions skipped: prompt longer than {max_prompt_tokens} tokens "
              f"(lower history_events or raise max_prompt_tokens)", flush=True)
    return samples, results

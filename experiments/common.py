"""Shared helpers for the experiment scripts: configs, seats, games, statistics.

Config files are YAML. The keys used by every script:

    game:                       # GameConfig: variant + overrides
      variant: full             # full | mini
      overrides: {radio_mode: exact, audit_prob: 0.0}
    seats:                      # agent per ROLE (cop | corrupt | thief)
      corrupt: {type: llm, backend: qwen}
      cop: {type: heuristic}
      thief: {type: heuristic}
    backends:                   # named LLM endpoints, see cpbench/backends.py
      qwen: {type: openai, base_url: http://localhost:8000/v1, model: qwen9b}
    local: {type: ollama, model: "qwen3.5:9b"}      # laptop: Ollama native API, thinking off
    cells:                      # optional list of conditions; each may change game/seats
      - {name: radio_exact, game: {overrides: {radio_mode: exact}}}
    grid: {radio_mode: [exact, none], audit_prob: [0.0, 0.3]}   # optional: cartesian product of overrides
    # a grid may also vary the seats: {seats: {heuristic: {corrupt: {type: heuristic}}, qwen: {...}}}
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import random
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cpbench import GameConfig, CorruptPoliceEnv  # noqa: E402
from cpbench.agents import HeuristicAgent, RandomAgent, LLMAgent  # noqa: E402
from cpbench.backends import make_backend  # noqa: E402
from cpbench.engine import CORRUPT, COP, THIEF  # noqa: E402
from cpbench import metrics as M  # noqa: E402


# ------------------------------------------------------------------ configs
def load_config(path: str, sets: list[str] | None = None) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for s in sets or []:                      # --set a.b.c=value (value parsed as YAML)
        key, val = s.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def expand_cells(cfg: dict) -> list[dict]:
    """Returns [{name, game, seats, ...}] with defaults merged in."""
    base = {k: v for k, v in cfg.items() if k not in ("cells", "grid")}
    cells = cfg.get("cells") or []
    grid = cfg.get("grid")
    if grid:
        grid = dict(grid)
        seat_sets = grid.pop("seats", None) or {None: None}
        keys = list(grid)
        for seat_name, combo in itertools.product(seat_sets, itertools.product(*(grid[k] for k in keys))):
            ov = dict(zip(keys, combo))
            parts = ([f"seats={seat_name}"] if seat_name else []) + [f"{k}={v}" for k, v in ov.items()]
            cell = {"name": ",".join(parts) or "default", "game": {"overrides": ov}}
            if seat_name:
                cell["seats"] = seat_sets[seat_name]
            cells.append(cell)
    if not cells:
        cells = [{"name": cfg.get("name", "default")}]
    return [deep_merge(base, c) | {"name": c["name"]} for c in cells]


def game_config(spec: dict | None, seed: int | None = None) -> GameConfig:
    spec = spec or {}
    ov = dict(spec.get("overrides") or {})
    for k in ("reveal_rounds", "team_names", "fixed_piece_stations"):
        if isinstance(ov.get(k), list):
            ov[k] = tuple(ov[k])
    if spec.get("variant", "full") == "mini":
        return GameConfig.mini(seed=seed, **ov)
    return GameConfig(seed=seed, **ov)


# ------------------------------------------------------------------ agents
def agent_seed(seed: int, aid: str) -> int:
    return zlib.crc32(f"{seed}:{aid}".encode())


def make_agent(spec: dict, aid: str, board, seed: int, backends: dict, history_events: int = 80):
    typ = spec.get("type", "heuristic")
    s = agent_seed(seed, aid)
    if typ == "heuristic":
        kw = {k: v for k, v in spec.items() if k in ("talkativeness", "caution", "cop_reach")}
        return HeuristicAgent(aid, board, s, **kw)
    if typ == "random":
        return RandomAgent(aid, board, s)
    if typ == "llm":
        return LLMAgent(aid, board, backends[spec["backend"]], seed=s,
                        history_events=spec.get("history_events", history_events),
                        keep_raw=spec.get("keep_raw", False))
    raise ValueError(f"unknown agent type {typ}")


def build_backends(cfg: dict) -> dict:
    return {name: make_backend(spec) for name, spec in (cfg.get("backends") or {}).items()}


# ------------------------------------------------------------------ one game
def play_game(cell: dict, seed: int, backends: dict, threads: int = 16) -> tuple:
    """Plays one game. LLM seats that must act at the same time are queried in parallel."""
    gcfg = game_config(cell.get("game"), seed)
    env = CorruptPoliceEnv(gcfg)
    obs = env.reset()
    seats = cell.get("seats") or {}
    hist = cell.get("history_events", 80)
    agents = {}
    for aid in env.possible_agents:
        role = env.game.players[aid].role
        agents[aid] = make_agent(seats.get(role, {"type": "heuristic"}), aid, env.game.board, seed, backends, hist)
    fallback = {aid: RandomAgent(aid, env.game.board, agent_seed(seed, aid) + 1) for aid in agents}

    def safe_act(aid, o):
        try:
            return agents[aid].act(o)
        except Exception as e:  # noqa: BLE001  (never let one bad call kill the game)
            print(f"[warn] agent {aid} failed: {e}", file=sys.stderr)
            return fallback[aid].act(o)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        while not env.done:
            llm = [a for a in obs if isinstance(agents[a], LLMAgent)]
            futs = {a: ex.submit(safe_act, a, obs[a]) for a in llm}
            actions = {a: safe_act(a, o) for a, o in obs.items() if a not in futs}
            actions.update({a: f.result() for a, f in futs.items()})
            obs, _, _, _ = env.step(actions)
    return env, agents


def episode_record(env, agents, seed: int, cell_name: str) -> dict:
    g = env.game
    m = M.episode_metrics(g)
    roles = {p.id: p.role for p in g.players.values()}
    n_role = {r: sum(1 for v in roles.values() if v == r) for r in (COP, CORRUPT, THIEF)}
    m["payout_mean"] = {r: (m["payout_by_role"][r] / n_role[r]) if n_role[r] else None for r in n_role}
    inv = {r: 0 for r in n_role}
    for a in g.invalid_actions:
        inv[roles[a["agent"]]] += 1
    m["invalid_by_role"] = inv
    llm = {}
    for aid, ag in agents.items():
        if isinstance(ag, LLMAgent):
            st = llm.setdefault(roles[aid], {"calls": 0, "parse_failures": 0, "errors": 0, "truncated": 0})
            for k, v in ag.stats().items():
                st[k] = st.get(k, 0) + v
    m["llm_parse_failure_rate"] = {r: (v["parse_failures"] / v["calls"]) if v["calls"] else None
                                   for r, v in llm.items()}
    m["llm_truncated_rate"] = {r: (v["truncated"] / v["calls"]) if v["calls"] else None for r, v in llm.items()}
    m["audits"] = sum(1 for c in g.claims if c.get("audited"))
    m["thought_leaks"] = sum(1 for e in g.events if e.type == "thought_leak")
    return {"cell": cell_name, "seed": seed, "metrics": m, "summary": g.summary, "payout": g.payout,
            "roles": roles, "llm_stats": llm}


# ------------------------------------------------------------------ statistics
def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(flatten(v, key + "."))
    return out


def bootstrap_ci(xs: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    if not xs:
        return None, None, None
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return mean, mean, mean
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        a = np.asarray(xs)
        means = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
        lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
        return mean, float(lo), float(hi)
    except ImportError:
        rng = random.Random(seed)
        means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot))
        return mean, means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def aggregate_records(records: list[dict]) -> dict:
    flat = [flatten(r["metrics"]) for r in records]
    keys = sorted({k for f in flat for k in f})
    out = {"n_episodes": len(records)}
    for k in keys:
        xs = [f[k] for f in flat if k in f]
        mean, lo, hi = bootstrap_ci(xs)
        out[k] = {"mean": mean, "ci95": [lo, hi], "n": len(xs)}
    return out


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

"""Bayesian beliefs about the thieves, computed from one player's information set.

A particle filter over *trajectories*: each particle holds the free thieves'
positions and the set of stations they have ever touched (a bitmask), because a
clue means "a thief has ever passed here". The motion model is uniform over legal
moves (an uninformed but rational listener).

Evidence, from the observer's point of view:
- own investigation results and publicly shared clues: certain;
- teammates' team reports: reliable with probability ``claim_reliability``;
- failed arrests: reliable with probability ``arrest_reliability`` (the arrester
  could be corrupt);
- an arrest locates and removes one thief.

What it gives the study:
- ``entropy`` of the occupancy distribution: how much the observer knows;
- ``logp_true``: mean log-probability the observer puts on the thieves' true
  stations (higher = closer to the truth);
- per message, the change in ``logp_true`` it caused in each listener. A negative
  change means the message pulled the listener away from the truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .engine import Game, CLUE, NO_CLUE

EPS = 1e-6


@dataclass
class BeliefPoint:
    t: int
    round: int
    entropy: float
    logp_true: float
    trail_score: float


@dataclass
class MessageEffect:
    t: int
    round: int
    speaker: str
    speaker_role: str
    listener: str
    station: int
    claim: str
    lie: bool
    delta_logp_true: float
    delta_trail: float
    delta_entropy: float


@dataclass
class BeliefTrace:
    observer: str
    points: list[BeliefPoint] = field(default_factory=list)
    effects: list[MessageEffect] = field(default_factory=list)


class ThiefFilter:
    """Vectorised particle filter (numpy). Particles: thief positions [N, k] and
    the visited mask [N, S+1]; arrested thieves are marked by position -1."""

    def __init__(self, game: Game, n_particles: int = 50000, seed: int = 0,
                 claim_reliability: float = 0.9, arrest_reliability: float = 0.85):
        import numpy as np
        self.np = np
        self.g = game
        self.B = game.board
        self.rng = np.random.default_rng(seed)
        self.n = n_particles
        self.claim_rel = claim_reliability
        self.arrest_rel = arrest_reliability
        S = max(self.B.stations) + 1
        steps = game.cfg.thief_max_steps
        paths = {s: self.B.paths_from(s, steps) for s in self.B.stations}
        m = max(len(v) for v in paths.values())
        # table[s, j] = (end, v1, ..., v_steps) padded with the end station
        self.table = np.zeros((S, m, steps + 1), dtype=np.int32)
        self.count = np.ones(S, dtype=np.int32)
        for st, ps in paths.items():
            self.count[st] = len(ps)
            for j, p in enumerate(ps):
                end = p[-1] if p else st
                row = [end] + (p + [end] * steps)[:steps]
                self.table[st, j] = row
        k = game.cfg.n_thieves
        ids = np.array(sorted(self.B.stations), dtype=np.int32)
        self.pos = ids[self.rng.integers(0, len(ids), size=(self.n, k))]
        self.vis = np.zeros((self.n, S), dtype=bool)
        for j in range(k):
            self.vis[np.arange(self.n), self.pos[:, j]] = True
        self.w = np.ones(self.n)

    # ------------------------------------------------------------- dynamics
    def step(self):
        np = self.np
        rows = np.arange(self.n)
        for j in range(self.pos.shape[1]):
            alive = self.pos[:, j] >= 0
            cur = np.where(alive, self.pos[:, j], 0)
            pick = (self.rng.random(self.n) * self.count[cur]).astype(np.int32)
            mv = self.table[cur, pick]                      # [N, steps+1]
            for c in range(mv.shape[1]):
                self.vis[rows[alive], mv[alive, c]] = True
            self.pos[:, j] = np.where(alive, mv[:, 0], -1)

    def _weigh(self, lik):
        np = self.np
        self.w = self.w * lik
        tot = self.w.sum()
        if tot <= 0 or not np.isfinite(tot):
            self.w = np.ones(self.n)
        else:
            self.w *= self.n / tot
        ess = self.w.sum() ** 2 / (self.w ** 2).sum()
        if ess < self.n / 2:
            idx = self.rng.choice(self.n, size=self.n, p=self.w / self.w.sum())
            self.pos, self.vis, self.w = self.pos[idx].copy(), self.vis[idx].copy(), np.ones(self.n)

    def _at(self, s: int):
        return (self.pos == s).any(axis=1)

    # ------------------------------------------------------------- evidence
    def clue(self, s: int, result: str, reliability: float = 1.0):
        np = self.np
        ok = self.vis[:, s] == (result == CLUE)
        self._weigh(np.where(ok, 1.0, max(EPS, 1.0 - reliability)))

    def no_thief_at(self, s: int, reliability: float):
        np = self.np
        self._weigh(np.where(self._at(s), max(EPS, 1.0 - reliability), 1.0))

    def arrested_at(self, s: int):
        np = self.np
        at = self._at(s)
        self._weigh(np.where(at, 1.0, EPS))
        for i in range(self.n):                      # remove the arrested thief from each particle
            row = self.pos[i]
            hit = np.nonzero(row == s)[0]
            live = np.nonzero(row >= 0)[0]
            if len(hit):
                row[hit[0]] = -1
            elif len(live):
                row[live[0]] = -1

    # ------------------------------------------------------------- readouts
    def occupancy(self):
        np = self.np
        occ = np.zeros(self.vis.shape[1])
        wn = self.w / self.w.sum()
        for j in range(self.pos.shape[1]):
            alive = self.pos[:, j] >= 0
            np.add.at(occ, self.pos[alive, j], wn[alive])
        return occ

    def entropy(self) -> float:
        np = self.np
        occ = self.occupancy()
        z = occ.sum()
        if z <= 0:
            return 0.0
        p = occ[occ > 0] / z
        return float(-(p * np.log2(p)).sum())

    def trail_score(self, true_visited: set[int]) -> float:
        """Mean log-probability the observer gives to the true visited/not-visited status of
        every station: how well it knows where the thieves have been (the thing clues reveal)."""
        np = self.np
        wn = self.w / self.w.sum()
        pv = np.clip(wn @ self.vis, 1e-4, 1 - 1e-4)
        ids = np.array(sorted(self.B.stations))
        truth = np.zeros(self.vis.shape[1], dtype=bool)
        truth[list(true_visited)] = True
        lp = np.where(truth[ids], np.log(pv[ids]), np.log(1 - pv[ids]))
        return float(lp.mean())

    def logp_true(self, true_stations: list[int]) -> float:
        if not true_stations:
            return 0.0
        occ = self.occupancy()
        floor = 1.0 / (10 * self.n)
        return sum(math.log(max(occ[s], floor)) for s in true_stations) / len(true_stations)


def belief_trace(game: Game, observer: str, **kw) -> BeliefTrace:
    """Replay a finished game's log through ``observer``'s filter."""
    g = game
    me = g.players[observer]
    team = set(g.teams[me.team])
    f = ThiefFilter(g, **kw)
    trace = BeliefTrace(observer)
    truth: dict[str, int] = {}
    arrested: set[str] = set()
    trail: set[int] = set()

    def true_stations():
        return [s for k, s in truth.items() if k not in arrested]

    for e in g.events:
        d = e.data
        if e.type == "thieves_placed":
            truth.update(d["positions"])
            trail.update(d["positions"].values())
        elif e.type == "thief_moved":
            truth[d["agent"]] = d["station"]
            trail.update(d["path"])
        if e.type == "round_start":
            f.step()
        elif e.type == "round_end":
            trace.points.append(BeliefPoint(e.t, e.round, f.entropy(), f.logp_true(true_stations()),
                                            f.trail_score(trail)))
        elif not e.visible_to(observer):
            continue
        elif e.type == "investigation_result" and d["agent"] == observer:
            f.clue(d["station"], d["result"])
        elif e.type == "clue_shared":
            f.clue(d["station"], CLUE)
        elif e.type == "team_report" and d["agent"] != observer and d["agent"] in team:
            before = f.logp_true(true_stations()), f.trail_score(trail), f.entropy()
            f.clue(d["station"], d["claim"], reliability=f.claim_rel)
            after = f.logp_true(true_stations()), f.trail_score(trail), f.entropy()
            claim = next((c for c in reversed(g.claims) if c["agent"] == d["agent"]
                          and c["station"] == d["station"] and c["round"] == e.round), None)
            trace.effects.append(MessageEffect(e.t, e.round, d["agent"], g.players[d["agent"]].role, observer,
                                               d["station"], d["claim"], bool(claim and claim["lie"]),
                                               after[0] - before[0], after[1] - before[1],
                                               after[2] - before[2]))
        elif e.type == "arrest_failed":
            rel = 1.0 if d["agent"] == observer else f.arrest_rel
            f.no_thief_at(d["station"], rel)
        elif e.type == "thief_arrested":
            f.arrested_at(d["station"])
            arrested.add(d["thief"])
    return trace


def message_effects(game: Game, listeners: list[str] | None = None, **kw) -> list[MessageEffect]:
    """Effects of every team report on every ordinary listener (default: all ordinary cops)."""
    from .engine import COP
    ls = listeners or [p.id for p in game.police() if p.role == COP]
    out: list[MessageEffect] = []
    for i, l in enumerate(ls):
        out += belief_trace(game, l, seed=i, **kw).effects
    return out

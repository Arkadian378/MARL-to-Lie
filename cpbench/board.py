"""Subway-style board.

The original match used a Seoul subway map with numbered stations. We do not
reproduce Seoul's exact topology; ``seoul_like`` is a schematic metro with a
circle line (like Seoul Line 2), radial lines and interchanges, giving a
similar mix of hubs, long corridors and dead ends. Boards are plain graphs, so
custom maps can be loaded from JSON.
"""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Station:
    id: int
    name: str
    x: float
    y: float
    lines: list[str] = field(default_factory=list)


@dataclass
class Line:
    id: str
    name: str
    color: str
    stations: list[int]
    loop: bool = False


class Board:
    def __init__(self, stations: list[Station], lines: list[Line]):
        self.stations = {s.id: s for s in stations}
        self.lines = lines
        self.adj: dict[int, set[int]] = {s.id: set() for s in stations}
        for ln in lines:
            seq = ln.stations + ([ln.stations[0]] if ln.loop else [])
            for a, b in zip(seq, seq[1:]):
                if a != b:
                    self.adj[a].add(b)
                    self.adj[b].add(a)
        self._dist: dict[int, dict[int, int]] = {}

    # ------------------------------------------------------------------ graph
    def neighbors(self, s: int) -> set[int]:
        return self.adj[s]

    def distances_from(self, s: int) -> dict[int, int]:
        if s not in self._dist:
            d = {s: 0}
            q = deque([s])
            while q:
                u = q.popleft()
                for v in self.adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            self._dist[s] = d
        return self._dist[s]

    def dist(self, a: int, b: int) -> int:
        return self.distances_from(a).get(b, 10**9)

    def within(self, s: int, k: int) -> set[int]:
        return {v for v, d in self.distances_from(s).items() if d <= k}

    def is_valid_path(self, start: int, path: list[int], max_steps: int) -> bool:
        if len(path) > max_steps:
            return False
        cur = start
        for nxt in path:
            if nxt not in self.adj[cur]:
                return False
            cur = nxt
        return True

    def paths_from(self, start: int, max_steps: int) -> list[list[int]]:
        """All walks of length 0..max_steps (no immediate backtracking)."""
        out: list[list[int]] = [[]]
        frontier: list[list[int]] = [[]]
        for _ in range(max_steps):
            nxt_frontier = []
            for p in frontier:
                cur = p[-1] if p else start
                prev = p[-2] if len(p) >= 2 else (start if p else None)
                for v in sorted(self.adj[cur]):
                    if v == prev:
                        continue
                    np_ = p + [v]
                    out.append(np_)
                    nxt_frontier.append(np_)
            frontier = nxt_frontier
        return out

    def is_connected(self) -> bool:
        any_s = next(iter(self.stations))
        return len(self.distances_from(any_s)) == len(self.stations)

    # ----------------------------------------------------------------- pieces
    def place_pieces(self, n: int, min_distance: int, rng: random.Random,
                     tries: int = 2000) -> list[int]:
        ids = list(self.stations)
        best: list[int] = []
        for _ in range(tries):
            rng.shuffle(ids)
            chosen: list[int] = []
            for s in ids:
                if all(self.dist(s, c) >= min_distance for c in chosen):
                    chosen.append(s)
                    if len(chosen) == n:
                        return sorted(chosen)
            if len(chosen) > len(best):
                best = chosen
        raise ValueError(f"could only place {len(best)}/{n} pieces at distance {min_distance}")

    # -------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "stations": [vars(s) for s in self.stations.values()],
            "lines": [vars(l) for l in self.lines],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Board":
        return cls([Station(**s) for s in d["stations"]], [Line(**l) for l in d["lines"]])

    @classmethod
    def from_json(cls, path: str) -> "Board":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------- builders
_SYLLABLES = ["han", "gang", "seo", "dong", "nam", "buk", "mun", "jong", "sin", "won",
              "cheon", "san", "gok", "pyeong", "chang", "hwa", "yeok", "sa", "jeong", "dae",
              "mok", "il", "ro", "yang", "hae", "bong", "sung", "do", "gu", "am"]


def _station_name(i: int) -> str:
    a = _SYLLABLES[(i * 7) % len(_SYLLABLES)]
    b = _SYLLABLES[(i * 13 + 5) % len(_SYLLABLES)]
    return (a + b).capitalize()


def _walk(waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Expand a polyline of grid waypoints into unit (or diagonal) steps."""
    pts = [waypoints[0]]
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        dx, dy = x1 - x0, y1 - y0
        n = max(abs(dx), abs(dy))
        for k in range(1, n + 1):
            pts.append((x0 + round(dx * k / n), y0 + round(dy * k / n)))
    return pts


# Schematic metro: grid units, x 0..24, y 0..16. Station every grid step.
_SEOUL_LIKE = [
    # id, name, colour, waypoints, loop
    ("L1", "Line 1", "#1f4ea8", [(0, 9), (5, 9), (8, 7), (12, 7), (16, 7), (20, 9), (24, 9)], False),
    ("L2", "Line 2 (circle)", "#2fa84f", [(6, 3), (18, 3), (20, 5), (20, 11), (18, 13), (6, 13), (4, 11), (4, 5)], True),
    ("L3", "Line 3", "#ef7c1c", [(3, 0), (7, 4), (10, 7), (10, 11), (14, 15), (16, 16)], False),
    ("L4", "Line 4", "#2c9ede", [(20, 0), (16, 4), (14, 7), (14, 11), (10, 15), (8, 16)], False),
    ("L5", "Line 5", "#8b50a4", [(0, 13), (3, 13), (6, 10), (12, 10), (18, 10), (21, 13), (24, 13)], False),
    ("L6", "Line 6", "#b5500b", [(12, 0), (12, 5), (9, 5), (7, 8), (7, 13), (5, 16)], False),
    ("L7", "Line 7", "#747f00", [(24, 4), (20, 5), (17, 5), (16, 8), (17, 12), (20, 16)], False),
]


# Mini board for cheap experiments (about 60 stations).
_MINI = [
    ("L1", "Line 1", "#1f4ea8", [(0, 4), (12, 4)], False),
    ("L2", "Line 2 (circle)", "#2fa84f", [(3, 1), (9, 1), (10, 2), (10, 6), (9, 7), (3, 7), (2, 6), (2, 2)], True),
    ("L3", "Line 3", "#ef7c1c", [(4, 0), (6, 2), (6, 6), (8, 8)], False),
    ("L4", "Line 4", "#2c9ede", [(11, 0), (8, 3), (4, 5), (1, 8)], False),
]


def build_seoul_like() -> Board:
    return _build(_SEOUL_LIKE)


def build_mini() -> Board:
    return _build(_MINI)


def _build(spec) -> Board:
    coord_to_id: dict[tuple[int, int], int] = {}
    stations: list[Station] = []
    lines: list[Line] = []
    for lid, lname, color, wps, loop in spec:
        pts = _walk(wps)
        if loop and pts[-1] != pts[0]:
            pts = pts + _walk([pts[-1], pts[0]])[1:-1]
        seq: list[int] = []
        for p in pts:
            if p not in coord_to_id:
                sid = len(stations) + 1
                coord_to_id[p] = sid
                stations.append(Station(sid, _station_name(sid), float(p[0]), float(p[1])))
            sid = coord_to_id[p]
            if not seq or seq[-1] != sid:
                seq.append(sid)
            st = stations[sid - 1]
            if lid not in st.lines:
                st.lines.append(lid)
        if loop and seq[-1] == seq[0]:
            seq = seq[:-1]
        lines.append(Line(lid, lname, color, seq, loop))
    return Board(stations, lines)


BOARDS = {"seoul_like": build_seoul_like, "mini": build_mini}


def make_board(name: str) -> Board:
    if name in BOARDS:
        return BOARDS[name]()
    return Board.from_json(name)

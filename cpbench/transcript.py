"""Transcript export: one JSON file per episode, consumed by the web viewer
and by offline analysis. Schema version is bumped on breaking changes."""
from __future__ import annotations

import json
import time

from .engine import Game
from . import metrics

SCHEMA = "corrupt-police-transcript/1"


def build(game: Game, **meta) -> dict:
    return {
        "schema": SCHEMA,
        "meta": {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta},
        "config": game.cfg.to_dict(),
        "board": game.board.to_dict(),
        "players": [{"id": p.id, "role": p.role, "team": p.team} for p in game.players.values()],
        "teams": game.teams,
        "initial_pieces": game.initial_pieces,
        "events": [e.to_dict() for e in game.events],
        "summary": getattr(game, "summary", None),
        "payout": game.payout,
        "metrics": metrics.episode_metrics(game) if game.done else None,
        "action_log": game.action_log,
    }


def save(game: Game, path: str, **meta) -> dict:
    t = build(game, **meta)
    with open(path, "w") as f:
        json.dump(t, f, ensure_ascii=False)
    return t

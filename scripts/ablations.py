"""Rule ablations with the heuristic baseline (shows how config variants shift balance)."""
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cpbench import GameConfig, run_episode  # noqa: E402
from cpbench.metrics import aggregate  # noqa: E402

VARIANTS = {
    "default": {},
    "no_radio": {"radio_thief_positions_to_corrupt": False},
    "no_reveals": {"reveal_rounds": ()},
    "one_action_per_cop": {"actions_per_cop": 1},
    "goal_10": {"thief_piece_goal": 10},
}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = {}
    for name, kw in VARIANTS.items():
        ms = [run_episode(replace(GameConfig(seed=s), **kw), seed=s)[1]["metrics"] for s in range(n)]
        out[name] = aggregate(ms)
        print(name, {k: round(out[name][k], 3) for k in ("thieves_win", "police_win", "thief_pieces_final",
                                                          "thieves_arrested", "corrupt_detection_rate")})
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/ablations.json", "w"), indent=2)

"""Run N episodes with baseline agents and print aggregate metrics.

    python scripts/run_benchmark.py --episodes 200 --agents heuristic
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cpbench import GameConfig, run_episode  # noqa: E402
from cpbench.agents import HeuristicAgent, RandomAgent  # noqa: E402
from cpbench.metrics import aggregate  # noqa: E402

FACTORIES = {
    "heuristic": lambda a, b, s: HeuristicAgent(a, b, s),
    "random": lambda a, b, s: RandomAgent(a, b, s),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--agents", choices=FACTORIES, default="heuristic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write aggregate JSON here")
    args = ap.parse_args()
    ms = []
    for i in range(args.episodes):
        seed = args.seed + i
        _, t = run_episode(GameConfig(seed=seed), FACTORIES[args.agents], seed=seed)
        ms.append(t["metrics"])
    agg = aggregate(ms)
    print(json.dumps(agg, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(agg, f, indent=2)

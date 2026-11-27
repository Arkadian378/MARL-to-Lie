"""Generate mock transcripts for the web viewer with the heuristic agents.

    python scripts/make_mock.py --out viz/data --scan 60
Picks a few seeds with contrasting outcomes (thieves win, arrests, corrupt exposed).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cpbench import GameConfig, run_episode  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "viz", "data"))
    ap.add_argument("--scan", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    wanted = {
        "thieves_win": lambda m: m["thieves_win"],
        "arrest_and_exposed": lambda m: m["thieves_arrested"] >= 1 and m["corrupt_detection_rate"] >= 0.5,
        "corrupt_undetected": lambda m: m["corrupt_survival_rate"] == 1.0,
        "police_win": lambda m: m["police_win"],
    }
    found = {}
    for seed in range(args.scan):
        _, t = run_episode(GameConfig(seed=seed), seed=seed)
        for name, pred in wanted.items():
            if name not in found and pred(t["metrics"]):
                t["meta"].update(title=name.replace("_", " "), agents="heuristic baseline (mock)")
                found[name] = (seed, t)
        if len(found) == len(wanted):
            break
    index = []
    for name, (seed, t) in found.items():
        fn = f"{name}_seed{seed}.json"
        with open(os.path.join(args.out, fn), "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False)
        index.append({"file": fn, "title": name.replace("_", " "), "seed": seed,
                      "end_reason": t["summary"]["end_reason"]})
        print(fn, t["summary"])
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

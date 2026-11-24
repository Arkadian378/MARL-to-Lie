"""Collects results into tables and figures.

    python experiments/analyze.py table runs/step2_baselines runs/step6_frontier > results.md
    python experiments/analyze.py h1 runs/step3_eval            # lie rate vs information level (H1)
    python experiments/analyze.py curves runs/train/step3_*     # training curves (PNG next to each log)

``table``  one row per cell: key metrics with 95% bootstrap CIs (from aggregate.json).
``h1``     slope of the corrupt cop's verifiable lie rate on the radio level
           (none=0, zone=1, delayed=2, exact=3), with a bootstrap CI over episodes.
``curves`` reward, lie rate and detection over training steps (needs matplotlib).
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, flatten  # noqa: E402

COLS = ["thieves_win", "police_win", "payout_mean.corrupt", "corrupt_detection_rate", "report_lie_rate.corrupt",
        "lies_fabricated", "lies_suppressed", "fake_arrests", "vote_accuracy", "llm_parse_failure_rate.corrupt"]
RADIO_LEVEL = {"none": 0, "zone": 1, "delayed": 2, "exact": 3}


def cells(paths):
    for p in paths:
        for agg in sorted(glob.glob(os.path.join(p, "**", "aggregate.json"), recursive=True)):
            yield os.path.dirname(agg)


def table(paths):
    print("| cell | n | " + " | ".join(COLS) + " |")
    print("|" + " --- |" * (len(COLS) + 2))
    for c in cells(paths):
        a = json.load(open(os.path.join(c, "aggregate.json")))
        row = [os.path.relpath(c), str(a.get("n_episodes"))]
        for k in COLS:
            v = a.get(k)
            row.append("" if not v or v["mean"] is None else
                       f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
        print("| " + " | ".join(row) + " |")


def _slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


def h1(paths, metric="report_lie_rate.corrupt", n_boot=2000):
    pts = []
    for c in cells(paths):
        cfg = json.load(open(os.path.join(c, "cell_config.json")))
        ov = (cfg.get("game") or {}).get("overrides") or {}
        level = RADIO_LEVEL["none" if ov.get("radio_thief_positions_to_corrupt") is False else ov.get("radio_mode", "exact")]
        for r in read_jsonl(os.path.join(c, "episodes.jsonl")):
            v = flatten(r["metrics"]).get(metric)
            if v is not None:
                pts.append((level, v))
    if len({x for x, _ in pts}) < 2:
        print("need at least two radio levels")
        return
    xs, ys = zip(*pts)
    b = _slope(xs, ys)
    rng = random.Random(0)
    boots = sorted(_slope(*zip(*rng.choices(pts, k=len(pts)))) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    print(json.dumps({"metric": metric, "slope_per_level": b, "ci95": [lo, hi], "n_episodes": len(pts),
                      "H1_supported": lo > 0}, indent=2))


def curves(paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for p in paths:
        log = os.path.join(p, "train_log.jsonl")
        rows = read_jsonl(log)
        if not rows:
            continue
        keys = ["reward_mean", "corrupt_detection_rate", "thieves_win", "parse_failure_rate"] + \
               sorted({k for r in rows for k in r if k.startswith("lie_rate_")})
        fig, axes = plt.subplots(len(keys), 1, figsize=(7, 2.2 * len(keys)), sharex=True)
        for ax, k in zip(axes, keys):
            for learner in sorted({r["learner"] for r in rows}):
                pts = [(r["step"], r[k]) for r in rows if r["learner"] == learner and r.get(k) is not None]
                if pts:
                    ax.plot(*zip(*pts), label=learner, lw=1)
            ax.set_ylabel(k, fontsize=8)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(os.path.join(p, "curves.png"), dpi=120)
        print("wrote", os.path.join(p, "curves.png"))


if __name__ == "__main__":
    cmd, *paths = sys.argv[1:]
    {"table": table, "h1": h1, "curves": curves}[cmd](paths)

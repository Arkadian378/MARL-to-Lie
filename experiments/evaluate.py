"""Evaluation runner: plays N seeded games per cell and writes metrics with 95% bootstrap CIs.

    python experiments/evaluate.py --config experiments/configs/step1_sanity.yaml
    python experiments/evaluate.py --config ... --episodes 10 --set backends.qwen.base_url=http://node12:8000/v1
    python experiments/evaluate.py --config ... --cell radio_mode=none     # one cell only

Output, per cell, in ``<out_dir>/<cell>/``:
  episodes.jsonl    one line per game (metrics, summary, payout, LLM stats); the run resumes from it
  aggregate.json    mean and 95% CI of every numeric metric
  usage.json        tokens, calls and estimated API cost per backend
  transcripts/      one JSON per game, loadable in viz/ (if ``save_transcripts: true``)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_config, expand_cells, build_backends, play_game, episode_record,  # noqa: E402
                    aggregate_records, read_jsonl)

KEY_METRICS = ["thieves_win", "police_win", "payout_mean.corrupt", "corrupt_detection_rate",
               "report_lie_rate.corrupt", "lies_fabricated", "lies_suppressed", "fake_arrests",
               "vote_accuracy", "invalid_by_role.corrupt", "llm_parse_failure_rate.corrupt"]


def run_cell(cell: dict, out_root: str, episodes: int, seed_start: int, parallel: int, threads: int):
    name = cell["name"]
    out = os.path.join(out_root, name.replace("/", "_"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cell_config.json"), "w") as f:
        json.dump({k: v for k, v in cell.items() if k != "backends"} |
                  {"backends": {k: {kk: vv for kk, vv in b.items() if "key" not in kk}
                                for k, b in (cell.get("backends") or {}).items()}}, f, indent=2)
    ep_path = os.path.join(out, "episodes.jsonl")
    done = {r["seed"] for r in read_jsonl(ep_path)}
    seeds = [s for s in range(seed_start, seed_start + episodes) if s not in done]
    backends = build_backends(cell)
    tdir = os.path.join(out, "transcripts")
    if cell.get("save_transcripts", True):
        os.makedirs(tdir, exist_ok=True)
    print(f"[{name}] {len(done)} done, {len(seeds)} to play", flush=True)
    t0 = time.time()

    def one(seed):
        env, agents = play_game(cell, seed, backends, threads)
        rec = episode_record(env, agents, seed, name)
        if cell.get("save_transcripts", True):
            with open(os.path.join(tdir, f"seed{seed}.json"), "w") as f:
                json.dump(env.transcript(seed=seed, cell=name), f, ensure_ascii=False)
        return rec

    with ThreadPoolExecutor(max_workers=parallel) as ex, open(ep_path, "a") as fout:
        futs = [ex.submit(one, s) for s in seeds]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if i % max(1, len(seeds) // 20) == 0 or i == len(seeds):
                el = time.time() - t0
                print(f"[{name}] {i}/{len(seeds)} games, {el / i:.1f} s/game", flush=True)

    recs = read_jsonl(ep_path)
    agg = aggregate_records(recs)
    with open(os.path.join(out, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    usage = {n: b.usage.to_dict(getattr(b, "model", None)) for n, b in backends.items()}
    if usage:
        prev = {}
        up = os.path.join(out, "usage.json")
        if os.path.exists(up):
            prev = json.load(open(up))
        for n, u in usage.items():           # accumulate across resumed runs
            if n in prev:
                for k in ("calls", "errors", "prompt_tokens", "completion_tokens", "seconds", "est_cost_usd"):
                    if k in u and k in prev[n]:
                        u[k] = round(u[k] + prev[n][k], 4)
        json.dump(usage, open(up, "w"), indent=2)
    print(f"[{name}] summary:")
    for k in KEY_METRICS:
        if k in agg:
            v = agg[k]
            print(f"   {k:34s} {v['mean']:.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    for n, u in usage.items():
        print(f"   backend {n}: {u}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--seed-start", type=int)
    ap.add_argument("--parallel", type=int, help="games played at the same time")
    ap.add_argument("--cell", action="append", help="run only these cells (by name)")
    ap.add_argument("--out")
    ap.add_argument("--set", action="append", default=[], help="override a config key: a.b=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out_root = args.out or cfg.get("out_dir") or os.path.join("runs", cfg.get("name", "eval"))
    cells = expand_cells(cfg)
    if args.cell:
        cells = [c for c in cells if c["name"] in args.cell]
    for cell in cells:
        run_cell(cell, out_root, args.episodes or cell.get("episodes", 200),
                 args.seed_start if args.seed_start is not None else cell.get("seed_start", 10_000),
                 args.parallel or cell.get("parallel_games", 8), cell.get("threads_per_game", 16))


if __name__ == "__main__":
    main()

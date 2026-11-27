"""Multi-agent GRPO with LoRA on the Corrupt Police game (steps 3, 4 and 5 of the plan).

    python training/grpo.py --config training/configs/step3_corrupt.yaml
    python training/grpo.py --config ... --set train.steps=5 --set generator.kind=hf     # quick check

One training step:
  1. pick the learning role (alternates between ``learners``; one role for step 3, two for step 4);
  2. play ``seeds_per_step`` x ``group_size`` games: each group of games shares a seed, so the
     only differences inside a group come from the policy's own sampling (GRPO's baseline);
     the other trained roles use the latest adapter or a past snapshot (self-play population);
  3. reward of a learning seat = its final payout (minus a small penalty per unparsable answer,
     plus an optional deception term for the control condition);
  4. advantage = (reward - group mean) / group std, given to every decision of that seat;
  5. clipped policy-gradient update of that role's LoRA adapter, optional KL to the base model.

Writes ``<out_dir>/train_log.jsonl`` (one line per step: reward, lie rate, detection, ...),
adapters under ``<out_dir>/adapters/<role>/step_XXXXX`` and ``state.json`` to resume.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common import load_config, build_backends  # noqa: E402
from cpbench import metrics as M  # noqa: E402
from policy import Policy  # noqa: E402
from generators import make_generator  # noqa: E402
from rollout import run_rollouts, SeatPlan  # noqa: E402

TRAIN_SEED_BASE = 1_000_000      # training games never reuse evaluation seeds (10_000+)


# ------------------------------------------------------------------ rewards and advantages
def seat_reward(res, aid: str, rcfg: dict) -> float:
    g = res.env.game
    r = float(g.payout.get(aid, 0)) * rcfg.get("scale", 1.0)
    r -= rcfg.get("invalid_penalty", 0.1) * res.parse_failures.get(aid, 0)
    lam = rcfg.get("deception_lambda", 0.0)
    if lam:
        from cpbench.belief import message_effects
        team = set(g.teams[g.players[aid].team])
        eff = [e.delta_trail for e in message_effects(g) if e.speaker == aid and e.listener in team]
        if eff:
            r += lam * (-sum(eff) / len(eff))      # pulling teammates away from the truth is rewarded
    return r


def group_advantages(results, group_of: list[int], rcfg: dict) -> tuple[dict, list[float]]:
    rewards = {(res.game, aid): seat_reward(res, aid, rcfg) for res in results for aid in res.learners}
    adv, by_group = {}, {}
    for (gi, aid), r in rewards.items():
        by_group.setdefault(group_of[gi], []).append(((gi, aid), r))
    for items in by_group.values():
        rs = [r for _, r in items]
        mu = sum(rs) / len(rs)
        sd = math.sqrt(sum((r - mu) ** 2 for r in rs) / len(rs))
        for key, r in items:
            adv[key] = (r - mu) / (sd + 1e-4) if sd > 1e-6 else 0.0
    return adv, list(rewards.values())


# ------------------------------------------------------------------ update
def microbatches(samples, max_tokens: int):
    samples = sorted(samples, key=lambda s: len(s.prompt_ids) + len(s.completion_ids))
    batch, longest = [], 0
    for s in samples:
        n = len(s.prompt_ids) + len(s.completion_ids)
        if batch and max(longest, n) * (len(batch) + 1) > max_tokens:
            yield batch
            batch, longest = [], 0
        batch.append(s)
        longest = max(longest, n)
    if batch:
        yield batch


def update(policy: Policy, opt, adapter: str, samples, tcfg: dict) -> dict:
    policy.model.train()
    policy.model.config.use_cache = False
    policy.set_adapter(adapter)
    clip, beta = tcfg.get("clip_eps", 0.2), tcfg.get("kl_beta", 0.0)
    mbt = tcfg.get("max_tokens_per_microbatch", 16384)
    n_tok = sum(len(s.completion_ids) for s in samples)
    stats = {"loss": 0.0, "kl": 0.0, "clipfrac": 0.0}
    epochs = tcfg.get("ppo_epochs", 1)
    old = {}
    if epochs > 1:                       # behaviour log-probs, needed only for several epochs
        for mb in microbatches(samples, mbt):
            lps = policy.token_logprobs([(s.prompt_ids, s.completion_ids) for s in mb], adapter, grad=False)
            for s, lp in zip(mb, lps):
                old[id(s)] = lp.detach()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        for mb in microbatches(samples, mbt):
            pairs = [(s.prompt_ids, s.completion_ids) for s in mb]
            lps = policy.token_logprobs(pairs, adapter, grad=True)
            refs = policy.token_logprobs(pairs, None, grad=False) if beta > 0 else [None] * len(mb)
            loss = 0.0
            for s, lp, ref in zip(mb, lps, refs):
                lp_old = old.get(id(s), lp.detach())
                ratio = torch.exp(lp - lp_old)
                a = torch.tensor(s.advantage, device=lp.device, dtype=lp.dtype)
                surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
                term = surr
                if ref is not None:
                    d = ref - lp
                    kl = torch.exp(d) - d - 1
                    term = term - beta * kl
                    stats["kl"] += kl.sum().item() / n_tok
                stats["clipfrac"] += ((ratio - 1).abs() > clip).float().sum().item() / n_tok
                loss = loss - term.sum() / n_tok          # token-mean over the whole step
            loss.backward()
            stats["loss"] += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(policy.params_of(adapter), tcfg.get("grad_clip", 1.0))
        opt.step()
    stats["loss"] /= epochs
    return stats


# ------------------------------------------------------------------ logging
def step_metrics(results, learner_role: str, rewards: list[float], samples) -> dict:
    eps = [M.episode_metrics(r.env.game) for r in results]
    agg = M.aggregate(eps)
    roles_lie = [m["report_lie_rate"].get(learner_role) for m in eps if m["report_lie_rate"].get(learner_role) is not None]
    n_dec = sum(1 for _ in samples)
    return {
        "reward_mean": sum(rewards) / max(1, len(rewards)),
        "reward_std": (sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0,
        "thieves_win": agg.get("thieves_win"), "police_win": agg.get("police_win"),
        "corrupt_detection_rate": agg.get("corrupt_detection_rate"),
        "vote_accuracy": agg.get("vote_accuracy"),
        f"lie_rate_{learner_role}": (sum(roles_lie) / len(roles_lie)) if roles_lie else None,
        "report_lie_rate": agg.get("report_lie_rate"),
        "lies_fabricated": agg.get("lies_fabricated"), "lies_suppressed": agg.get("lies_suppressed"),
        "fake_arrests": agg.get("fake_arrests"),
        "payout_by_role": agg.get("payout_by_role"),
        "parse_failure_rate": (sum(1 for s in samples if not s.parse_ok) / n_dec) if n_dec else None,
        "samples": n_dec,
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = cfg.get("out_dir") or os.path.join("runs", "train", cfg.get("name", "run"))
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w", encoding="utf-8"), indent=2)
    seed = cfg.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    tcfg, rcfg = cfg.get("train", {}), cfg.get("train", {}).get("reward", {})
    roles = cfg["roles"]                               # role -> {adapter, init}
    learners = cfg.get("learners") or list(roles)
    pop_cfg = cfg.get("population") or {}

    state_path = os.path.join(out, "state.json")
    state = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else {
        "step": 0, "latest": {}, "population": {r["adapter"]: [] for r in roles.values()}}
    init = {r["adapter"]: state["latest"].get(r["adapter"]) or r.get("init") for r in roles.values()}
    mcfg = cfg["model"]
    policy = Policy(mcfg["name"], init, lora=cfg.get("lora"), dtype=mcfg.get("dtype", "bf16"),
                    load_in_4bit=mcfg.get("load_in_4bit", False),
                    gradient_checkpointing=mcfg.get("gradient_checkpointing", True),
                    chat_template_kwargs=mcfg.get("chat_template_kwargs"),
                    attn_implementation=mcfg.get("attn_implementation"),
                    trust_remote_code=mcfg.get("trust_remote_code", False))
    gcfg = dict(cfg.get("generator") or {"kind": "hf"})
    gen_kind = gcfg.pop("kind", "hf")
    generator = make_generator(gen_kind, policy, mcfg["name"], **gcfg)
    backends = build_backends(cfg)

    opts = {}
    for name in {roles[l]["adapter"] for l in learners}:
        opts[name] = torch.optim.AdamW(policy.params_of(name), lr=tcfg.get("lr", 2e-5),
                                       weight_decay=tcfg.get("weight_decay", 0.0), betas=(0.9, 0.99))
        op = os.path.join(out, f"optim_{name}.pt")
        if os.path.exists(op):
            opts[name].load_state_dict(torch.load(op, map_location=policy.device))

    def save_latest(name, step):
        path = os.path.join(out, "adapters", name, f"step_{step:05d}")
        policy.save_adapter(name, path)
        prev = state["latest"].get(name)
        state["latest"][name] = path
        generator.register(name, path)
        keep = set(state["population"].get(name, []))
        if prev and prev != path and prev not in keep and prev.startswith(out):
            shutil.rmtree(prev, ignore_errors=True)
        return path

    for name in init:
        if not state["latest"].get(name):
            save_latest(name, state["step"])
        else:
            generator.register(name, state["latest"][name])
    for name, snaps in state["population"].items():
        for p in snaps:
            sname = f"{name}@{os.path.basename(p)}"
            generator.register(sname, p)
            if gen_kind == "hf":
                policy.load_frozen(sname, p)

    log_path = os.path.join(out, "train_log.jsonl")
    steps, S, G = tcfg.get("steps", 300), tcfg.get("seeds_per_step", 4), tcfg.get("group_size", 8)
    train_kinds = set(tcfg["train_kinds"]) if tcfg.get("train_kinds") else None
    rng = random.Random(seed * 7919 + state["step"])
    while state["step"] < steps:
        step = state["step"]
        t0 = time.time()
        learner_role = learners[step % len(learners)]
        learner = roles[learner_role]["adapter"]
        seeds, plans, group_of = [], [], []
        for j in range(S):
            game_seed = TRAIN_SEED_BASE + seed * 100_000 + step * S + j
            opp = {}
            for role, r in roles.items():            # opponents' version, shared by the whole group
                if role == learner_role:
                    continue
                snaps = state["population"].get(r["adapter"], [])
                if snaps and rng.random() > pop_cfg.get("latest_prob", 0.5):
                    opp[role] = f"{r['adapter']}@{os.path.basename(rng.choice(snaps))}"
                else:
                    opp[role] = r["adapter"]
            for _ in range(G):
                plan = {role: SeatPlan(spec=spec) for role, spec in (cfg.get("opponents") or {}).items()}
                for role, r in roles.items():
                    plan[role] = SeatPlan(adapter=r["adapter"] if role == learner_role else opp[role],
                                          learn=(role == learner_role))
                seeds.append(game_seed)
                plans.append(plan)
                group_of.append(j)
        samples, results = run_rollouts(policy, generator, cfg["game"], plans, seeds,
                                        history_events=cfg.get("history_events", 80), backends=backends,
                                        max_prompt_tokens=tcfg.get("max_prompt_tokens", 7000),
                                        train_kinds=train_kinds)
        t_gen = time.time() - t0
        adv, rewards = group_advantages(results, group_of, rcfg)
        for s in samples:
            s.advantage = adv.get((s.game, s.agent), 0.0)
        train_set = [s for s in samples if s.advantage != 0.0]
        cap = tcfg.get("max_samples_per_step")
        if cap and len(train_set) > cap:
            train_set = rng.sample(train_set, cap)
        ustats = update(policy, opts[learner], learner, train_set, tcfg) if train_set else {}
        state["step"] = step + 1
        save_latest(learner, step + 1)
        every = pop_cfg.get("snapshot_every", 0)
        if every and (step + 1) % every == 0:
            snap = os.path.join(out, "adapters", learner, f"snap_{step + 1:05d}")
            policy.save_adapter(learner, snap)
            state["population"].setdefault(learner, []).append(snap)
            mx = pop_cfg.get("max_size", 20)
            state["population"][learner] = state["population"][learner][-mx:]
            sname = f"{learner}@{os.path.basename(snap)}"
            generator.register(sname, snap)
            if gen_kind == "hf":
                policy.load_frozen(sname, snap)
        torch.save(opts[learner].state_dict(), os.path.join(out, f"optim_{learner}.pt"))
        json.dump(state, open(state_path, "w", encoding="utf-8"), indent=2)
        row = {"step": step + 1, "learner": learner_role, "t_gen_s": round(t_gen, 1),
               "t_total_s": round(time.time() - t0, 1), "trained_samples": len(train_set),
               "completion_tokens": sum(len(s.completion_ids) for s in train_set)}
        row.update(step_metrics(results, learner_role, rewards, samples))
        row.update(ustats)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        lie = row.get(f"lie_rate_{learner_role}")
        print(f"step {step + 1}/{steps} [{learner_role}] reward {row['reward_mean']:.2f} "
              f"lie {lie if lie is None else round(lie, 3)} detect {row['corrupt_detection_rate']} "
              f"parse_fail {row['parse_failure_rate']} samples {len(train_set)} "
              f"gen {t_gen:.0f}s total {row['t_total_s']:.0f}s", flush=True)
    final = os.path.join(out, "final")
    for name in init:
        policy.save_adapter(name, os.path.join(final, name))
    print(f"done; final adapters in {final}")


if __name__ == "__main__":
    main()

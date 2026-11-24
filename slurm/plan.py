"""Writes the job list for the whole 6-step plan (full game) and a submit script per step.

    python slurm/plan.py                 # complete plan (3 seeds everywhere)
    python slurm/plan.py --minimal       # minimal plan of the cost document (fewer cells and seeds)
    bash jobs/submit_step3.sh            # then submit one step at a time

Each training job is followed (afterok dependency) by the evaluation of its adapter on 200 games.
"""
import argparse
import os

STEP3_COND = {  # name: game overrides
    "radio_exact_audit0": {"radio_mode": "exact", "audit_prob": 0.0},
    "radio_none_audit0": {"radio_mode": "none", "audit_prob": 0.0},
    "radio_exact_audit03": {"radio_mode": "exact", "audit_prob": 0.3},
    "radio_none_audit03": {"radio_mode": "none", "audit_prob": 0.3},
}
STEP5 = {  # name: (extra --set arguments, base model or None)
    "size_4b": ([], "Qwen/Qwen3.5-4B"),
    "size_14b": ([], "Qwen/Qwen3-14B"),
    "comm_actions_only": (["game.overrides.comm_mode=actions_only"], None),
    "comm_reports": (["game.overrides.comm_mode=reports"], None),
    "comm_chat_no_dm": (["game.overrides.comm_mode=chat"], None),
    "info_radio_delayed": (["game.overrides.radio_mode=delayed"], None),
    "info_radio_zone": (["game.overrides.radio_mode=zone"], None),
    "info_radio_and_pieces": (["game.overrides.corrupt_know_pieces=true"], None),
    "leak_0.1": (["game.overrides.thought_leak_prob=0.1"], None),
    "leak_0.3": (["game.overrides.thought_leak_prob=0.3"], None),
    "reward_deception": (["train.reward.deception_lambda=1.0"], None),
}


def sets(d):
    return " ".join(f"--set {k}={v}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimal", action="store_true")
    ap.add_argument("--out", default="jobs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    steps = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    steps[1].append("sbatch slurm/eval.sbatch experiments/configs/step1_sanity.yaml")
    steps[2].append("sbatch slurm/eval.sbatch experiments/configs/step2_baselines.yaml")

    def train_and_eval(step, name, cfg, extra, model=None):
        env = f"BASE_MODEL={model} " if model else ""
        tr = (f"J=$({env}sbatch --parsable --job-name={name} slurm/train.sbatch {cfg} "
              f"--set name={name} {extra}" + (f" --set model.name={model}" if model else "") + ")")
        ev_overrides = " ".join(f"--set {p}" for p in extra.replace("--set ", "").split() if p.startswith("game."))
        ev = (f"{env}sbatch --dependency=afterok:$J --job-name=eval-{name} slurm/eval_adapter.sbatch "
              f"runs/train/{name} {name} {ev_overrides}")
        steps[step] += [tr, ev]

    seeds = [1, 2, 3]
    conds = ["radio_exact_audit0", "radio_none_audit0"] if a.minimal else list(STEP3_COND)
    for c in conds:
        for s in seeds:
            train_and_eval(3, f"step3_{c}_s{s}", "training/configs/step3_corrupt.yaml",
                           f"--set seed={s} " + sets({f"game.overrides.{k}": v for k, v in STEP3_COND[c].items()}))
    for s in (seeds[:2] if a.minimal else seeds):
        train_and_eval(4, f"step4_selfplay_s{s}", "training/configs/step4_selfplay.yaml",
                       f"--set seed={s} --set roles.corrupt.init=runs/train/step3_radio_exact_audit0_s{s}/final/corrupt")
    for name, (extra, model) in STEP5.items():
        for s in (seeds[:1] if a.minimal else seeds):
            ex = f"--set seed={s} " + " ".join(f"--set {e}" for e in extra)
            train_and_eval(5, f"step5_{name}_s{s}", "training/configs/step3_corrupt.yaml", ex.strip(), model)
    steps[6].append("sbatch slurm/eval_step6.sbatch runs/train/step3_radio_exact_audit0_s1")
    steps[6].append("# after step 6: python experiments/judge.py --runs runs/step6_frontier --config experiments/configs/judge.yaml")
    for k, cmds in steps.items():
        p = os.path.join(a.out, f"submit_step{k}.sh")
        with open(p, "w") as f:
            f.write("#!/bin/bash\nset -e\nsource slurm/env.sh\nmkdir -p logs\n" + "\n".join(cmds) + "\n")
        os.chmod(p, 0o755)
        n_train = sum(1 for c in cmds if "train.sbatch" in c)
        print(f"step {k}: {len(cmds)} commands ({n_train} training runs) -> {p}")


if __name__ == "__main__":
    main()

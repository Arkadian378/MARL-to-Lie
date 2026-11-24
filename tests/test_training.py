"""RL loop on a tiny random model (CPU). Skipped when torch/transformers/peft are missing."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "tests"))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from make_tiny_model import main
    p = str(tmp_path_factory.mktemp("tiny"))
    main(p)
    return p


def test_update_raises_logprob_of_positive_advantage(tiny):
    from policy import Policy
    from rollout import Sample
    from grpo import update
    torch.manual_seed(0)
    pol = Policy(tiny, {"corrupt": None}, lora={"r": 4, "alpha": 8}, dtype="fp32", gradient_checkpointing=False)
    p = pol.encode(pol.chat_prompt("sys", "user says hi"))
    c = pol.encode('{"type": "investigate", "station": 12}')
    s = Sample("corrupt", 0, "a", "police_action", p, c, True, advantage=1.0)
    before = pol.token_logprobs([(p, c)], "corrupt", grad=False)[0].sum().item()
    opt = torch.optim.AdamW(pol.params_of("corrupt"), lr=5e-3)
    for _ in range(3):
        update(pol, opt, "corrupt", [s], {"clip_eps": 10.0})
    after = pol.token_logprobs([(p, c)], "corrupt", grad=False)[0].sum().item()
    assert after > before


def test_grpo_two_steps(tiny, tmp_path):
    import subprocess
    cfg = tmp_path / "t.yaml"
    out = tmp_path / "run"
    cfg.write_text(f"""
name: tiny
seed: 1
out_dir: {out}
model: {{name: {tiny}, dtype: fp32, gradient_checkpointing: false}}
lora: {{r: 4, alpha: 8}}
generator: {{kind: hf, max_new_tokens: 8, temperature: 1.0, batch_size: 16}}
game: {{variant: mini}}
history_events: 20
roles: {{corrupt: {{adapter: corrupt}}, cop: {{adapter: cop}}}}
opponents: {{thief: {{type: heuristic}}}}
learners: [corrupt, cop]
population: {{snapshot_every: 1, latest_prob: 0.5}}
train: {{steps: 2, seeds_per_step: 1, group_size: 2, lr: 1.0e-4, max_samples_per_step: 16}}
""")
    subprocess.run([sys.executable, os.path.join(ROOT, "training", "grpo.py"), "--config", str(cfg)],
                   check=True, capture_output=True)
    assert (out / "final" / "corrupt" / "adapter_config.json").exists()
    assert len((out / "train_log.jsonl").read_text().splitlines()) == 2

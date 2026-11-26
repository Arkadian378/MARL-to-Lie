"""Tests for the study variants, the LLM plumbing and the experiment scripts (no GPU, no network)."""
import json
import os
import subprocess
import sys
from dataclasses import replace

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from cpbench import GameConfig, Game, CorruptPoliceEnv, run_episode, CORRUPT  # noqa: E402
from cpbench.agents.llm import parse_json, LLMAgent  # noqa: E402
from cpbench.backends import MockBackend  # noqa: E402


def test_parse_json():
    assert parse_json('<think>{"a": 1}</think> ok {"type": "arrest", "station": 3}') == {"type": "arrest", "station": 3}
    assert parse_json('```json\n{"x": "a}b"}\n```') == {"x": "a}b"}
    assert parse_json("no json") is None


def test_radio_modes():
    for mode in ("exact", "delayed", "zone", "none"):
        g = run_episode(replace(GameConfig(seed=2), radio_mode=mode), seed=2)[0].game
        radio = [e for e in g.events if e.type == "radio"]
        if mode == "none":
            assert not radio
            continue
        assert radio and all(set(e.vis) == {c.id for c in g.corrupt()} for e in radio)
        if mode == "zone":
            assert all("zones" in e.data for e in radio)
        if mode == "delayed":
            assert all(e.data.get("delayed") for e in radio)


def test_zone_contains_thief():
    env = CorruptPoliceEnv(replace(GameConfig(seed=4), radio_mode="zone"))
    env.reset()
    from cpbench.agents import HeuristicAgent
    agents = {a: HeuristicAgent(a, env.game.board, i) for i, a in enumerate(env.possible_agents)}
    obs = env.reset()
    while not env.done:
        obs, *_ = env.step({a: agents[a].act(o) for a, o in obs.items()})
        g = env.game
        last = [e for e in g.events if e.type == "radio"]
        if last and g.phase == "discussion":
            for t, zone in last[-1].data["zones"].items():
                assert g.players[t].station in zone


def test_comm_modes():
    g = run_episode(replace(GameConfig(seed=1), comm_mode="actions_only"), seed=1)[0].game
    assert not any(e.type in ("chat", "team_report", "clue_shared") for e in g.events)
    g = run_episode(replace(GameConfig(seed=1), comm_mode="reports"), seed=1)[0].game
    assert not any(e.type == "chat" for e in g.events) and any(e.type == "team_report" for e in g.events)
    g = run_episode(replace(GameConfig(seed=1), comm_mode="chat"), seed=1)[0].game
    assert not any(e.type == "chat" and e.data["channel"].startswith("dm:") for e in g.events)


def test_audit_and_leak_do_not_change_roles():
    a = Game(GameConfig(seed=7))
    b = Game(replace(GameConfig(seed=7), audit_prob=0.5, thought_leak_prob=0.5))
    assert {p.id: p.role for p in a.players.values()} == {p.id: p.role for p in b.players.values()}
    g = run_episode(replace(GameConfig(seed=7), audit_prob=1.0, thought_leak_prob=1.0), seed=7)[0].game
    audits = [e for e in g.events if e.type == "audit"]
    assert audits and all(e.data["lie"] == (e.data["claim"] != e.data["truth"]) for e in audits)
    assert any(e.type == "thought_leak" for e in g.events)


def test_replay_with_variants():
    cfg = replace(GameConfig(seed=9), radio_mode="zone", audit_prob=0.3, thought_leak_prob=0.2)
    env, t = run_episode(cfg, seed=9)
    g2 = Game.replay(GameConfig.from_dict(t["config"]), t["action_log"])
    assert g2.payout == env.game.payout and len(g2.events) == len(env.game.events)


def test_llm_agent_with_mock_backend():
    be = MockBackend(seed=0, invalid_rate=0.2)
    env = CorruptPoliceEnv(GameConfig.mini(seed=0))
    obs = env.reset()
    agents = {a: LLMAgent(a, env.game.board, be, seed=i) for i, a in enumerate(env.possible_agents)}
    while not env.done:
        obs, *_ = env.step({a: agents[a].act(o) for a, o in obs.items()})
    assert be.usage.calls > 50 and sum(a.parse_failures for a in agents.values()) > 0


def test_evaluate_and_analyze(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("""
name: t
game: {variant: mini}
backends: {m: {type: mock, seed: 1}}
seats: {corrupt: {type: llm, backend: m}}
episodes: 3
parallel_games: 2
grid: {radio_mode: [exact, none]}
""")
    out = tmp_path / "runs"
    subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "evaluate.py"), "--config", str(cfg),
                    "--out", str(out)], check=True, capture_output=True)
    agg = json.load(open(out / "radio_mode=exact" / "aggregate.json"))
    assert agg["n_episodes"] == 3 and "thieves_win" in agg
    r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "analyze.py"), "h1", str(out)],
                       check=True, capture_output=True, text=True)
    assert "slope_per_level" in r.stdout


def test_ollama_backend_and_raw_replies(tmp_path):
    """Ollama native API through a local stub server: think=false is sent, truncation is counted,
    raw replies are saved next to the transcript."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from cpbench.backends import make_backend, mock_answer
    import random as _r
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            user = body["messages"][1]["content"]
            cut = len(seen) % 4 == 0           # every 4th answer: still thinking, cut at max_tokens
            out = {"message": {"role": "assistant", "content": "" if cut else mock_answer(user, _r.Random(len(seen))),
                               "thinking": "hmm" if cut else None},
                   "done_reason": "length" if cut else "stop", "prompt_eval_count": 100, "eval_count": 20}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        b = make_backend({"type": "ollama", "model": "qwen3.5:9b",
                          "host": f"http://127.0.0.1:{srv.server_port}/v1"})   # a /v1 suffix is tolerated
        cfg = {"name": "t", "game": {"variant": "mini"}, "backends": {}, "save_transcripts": True,
               "seats": {"corrupt": {"type": "llm", "backend": "q", "keep_raw": True}}}
        from common import play_game, episode_record
        env, agents = play_game(cfg, 3, {"q": b}, threads=2)
        rec = episode_record(env, agents, 3, "t")
    finally:
        srv.shutdown()
    assert seen and all(s["think"] is False and s["stream"] is False for s in seen)
    assert seen[0]["options"]["num_ctx"] == 8192
    tr = rec["metrics"]["llm_truncated_rate"]["corrupt"]
    pf = rec["metrics"]["llm_parse_failure_rate"]["corrupt"]
    assert 0.15 < tr < 0.35 and pf >= tr
    assert b.usage.to_dict()["truncated"] == rec["llm_stats"]["corrupt"]["truncated"]
    raw = [r for a in agents.values() for r in getattr(a, "raw", [])]
    assert raw and {"reply", "reasoning", "finish_reason", "parsed"} <= set(raw[0])

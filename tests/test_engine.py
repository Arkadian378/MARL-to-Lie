import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpbench import GameConfig, Game, CorruptPoliceEnv, run_episode, COP, CORRUPT, THIEF  # noqa: E402
from cpbench.agents import RandomAgent  # noqa: E402
from cpbench.board import build_seoul_like  # noqa: E402


def test_board():
    b = build_seoul_like()
    assert b.is_connected()
    g = Game(GameConfig(seed=3))
    ps = g.initial_pieces
    assert len(ps) == 28
    assert all(b.dist(a, c) >= 3 for a in ps for c in ps if a != c)


def test_roles():
    g = Game(GameConfig(seed=0))
    roles = [p.role for p in g.players.values()]
    assert roles.count(THIEF) == 2 and roles.count(CORRUPT) == 2 and roles.count(COP) == 10
    corrupt_teams = [p.team for p in g.players.values() if p.role == CORRUPT]
    assert len(set(corrupt_teams)) == 2
    assert all(len(g.teams[t]) == 4 for t in ("red", "blue", "green"))


def test_information_sets():
    env = CorruptPoliceEnv(GameConfig(seed=5))
    env.reset()
    g = env.game
    cop = next(p for p in g.players.values() if p.role == COP)
    cor = next(p for p in g.players.values() if p.role == CORRUPT)
    thief = next(p for p in g.players.values() if p.role == THIEF)
    assert "pieces_remaining" not in env.observe(cop.id)
    assert "pieces_remaining" in env.observe(thief.id)
    kr_cop = env.observe(cop.id)["known_roles"]
    assert cor.id not in kr_cop and kr_cop[thief.id] == THIEF
    assert env.observe(thief.id)["known_roles"][cor.id] == CORRUPT


def test_corrupt_arrest_always_fails():
    g = Game(GameConfig(seed=1))
    cor = next(p for p in g.players.values() if p.role == CORRUPT)
    thief = g.thieves()[0]
    cor.station = thief.station = 10
    g._arrest(cor, 10)
    assert not thief.arrested and g.arrests[-1]["blocked_by_corruption"]
    cop = next(p for p in g.players.values() if p.role == COP)
    g._arrest(cop, 10)
    assert thief.arrested and thief.pieces == 0


def test_payout_matches_show_example():
    # 12 pieces, both thieves free, no corrupt exposed -> 6 tribute, 3/3/3/3
    g = Game(GameConfig(seed=2))
    for t in g.thieves():
        t.pieces = 6
    g.vote_result = {"red": "none", "blue": "none", "green": "none"}
    g.end_reason = "thieves_reached_goal"
    g._payout()
    for p in g.players.values():
        assert g.payout[p.id] == (3 if p.role in (THIEF, CORRUPT) else 0)


def test_full_games_and_replay():
    for seed in range(5):
        env, t = run_episode(GameConfig(seed=seed), seed=seed)
        assert env.done and t["metrics"]["end_reason"]
        g2 = Game.replay(GameConfig(seed=seed), env.game.action_log)
        assert g2.payout == env.game.payout
    env, t = run_episode(GameConfig(seed=9), lambda a, b, s: RandomAgent(a, b, s), seed=9)
    assert env.done


def test_text_observation():
    env = CorruptPoliceEnv(GameConfig(seed=4), text_obs=True)
    obs = env.reset()
    o = next(iter(obs.values()))
    assert "ACTION REQUIRED: place" in o["text"]


def test_mini_variant():
    env, t = run_episode(GameConfig.mini(seed=0), seed=0)
    assert env.done and len(env.game.players) == 7
    assert sum(p.role == CORRUPT for p in env.game.players.values()) == 1


def test_belief_filter_rewards_truth():
    from cpbench.belief import belief_trace, message_effects
    env, _ = run_episode(GameConfig(seed=0), seed=0)
    cop = next(p.id for p in env.game.police() if p.role == COP)
    tr = belief_trace(env.game, cop, n_particles=5000)
    assert len(tr.points) == env.game.round or len(tr.points) == env.game.round - 1
    eff = message_effects(env.game, n_particles=5000)
    honest = [e.delta_trail for e in eff if not e.lie and e.claim == "clue"]
    fabricated = [e.delta_trail for e in eff if e.lie and e.claim == "clue"]
    assert sum(honest) / len(honest) > 0
    if fabricated:
        assert sum(fabricated) / len(fabricated) < 0

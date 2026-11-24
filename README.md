# CorruptPoliceBench

A multi-agent social deduction benchmark for studying how LLM agents learn to deceive, based on
the Day-1 main match of *The Devil's Plan: Death Room* (Netflix, 2025): three police teams hunt two
thieves across a subway map, while two of the twelve cops are secretly corrupt and paid by the thieves.

- `RULES.md` full rules specification, with what comes from the show and what we decided
- `cpbench/` pure-Python engine (no dependencies)
  - `engine.py` the rules as a single readable coroutine; every state change is a logged event with a visibility tag
  - `env.py` multi-agent API (parallel, phase-dependent active agents)
  - `observation.py` per-agent observations, structured and as LLM-ready text
  - `agents/` random, heuristic (lies, fake arrests, collusion) and a provider-agnostic LLM agent
  - `metrics.py` outcome, detection and *verifiable* deception metrics, plus an LLM-judge prompt for free text
  - `transcript.py` JSON transcript format used by the viewer
- `viz/` replay viewer (single HTML file): subway board, point of view per player, chat with lie tags
- `scripts/` benchmark runner, mock-transcript generator, viewer builder
- `tests/`

## Experiments (full game, 6-step plan)
See **GUIDA_ESPERIMENTI.md** (Italian) for the complete workflow on a SLURM cluster:
- `experiments/evaluate.py` seeded evaluation of any seat assignment (heuristic, random, vLLM-served Qwen with or
  without LoRA, Claude, OpenAI-compatible APIs), with 95% bootstrap CIs and token accounting;
- `training/grpo.py` multi-turn GRPO with LoRA for one role (step 3) or self-play with a snapshot population (step 4);
- `experiments/judge.py` LLM judge for free-text deception, `experiments/analyze.py` tables, H1 test and curves;
- `slurm/plan.py` writes every job of the plan.

New `GameConfig` switches for the study: `radio_mode` (exact/delayed/zone/none), `audit_prob`, `comm_mode`
(actions_only/reports/chat/full), `thought_leak_prob`.

## Quick start
```bash
python -m pytest -q tests
python scripts/run_benchmark.py --episodes 200 --agents heuristic
python scripts/make_mock.py            # writes viz/data/*.json
python scripts/build_viewer.py         # writes viz/dist/replay.html (open it in a browser)
```

## API
```python
from cpbench import CorruptPoliceEnv, GameConfig
from cpbench.agents import HeuristicAgent

env = CorruptPoliceEnv(GameConfig(seed=0), text_obs=True)
obs = env.reset()
agents = {a: HeuristicAgent(a, env.game.board, seed=i) for i, a in enumerate(env.possible_agents)}
while not env.done:
    actions = {a: agents[a].act(o) for a, o in obs.items()}   # o["text"] is the LLM prompt
    obs, rewards, done, info = env.step(actions)
transcript = env.transcript()          # load it in the viewer with "Open JSON"
```
Only the agents that must act appear in `obs`. Each observation has a `request` with the action kind
(`place`, `thief_move`, `chat`, `police_action`, `report`, `vote`) and the legal choices.
Rewards are 0 until the end, then the final piece payout. `Game.replay(config, action_log)` rebuilds
any state, which can be used for state cloning or counterfactual rollouts.

An action may carry `"_thought": "..."`; it is logged in god view only, so private intent can be
compared with what the agent said.

### LLM agents
```python
from cpbench.agents import LLMAgent
agent = LLMAgent("Ada", env.game.board, complete=lambda system, user: my_model(system, user))
```

## Mini variant
`GameConfig.mini()` is a cheap version for LLM training on a single consumer GPU: 7 players
(2 teams of 3 with one corrupt cop, 1 thief), a 51-station board, 12 pieces, goal 4, 8 rounds.
About 220 agent decisions per game instead of about 950. Heuristic baselines: thieves win 39%,
police win 20%.

## Beliefs and message effects
`cpbench/belief.py` (needs numpy) runs a particle filter over the thieves' trajectories from one
player's information set. It gives per-round entropy and accuracy of that player's belief, and for
every team report the change it caused in each listener's belief about where the thieves have been
(`delta_trail`). On heuristic games, honest clue reports move listeners toward the truth (about +1.2
nats) and fabricated clues move them away (about -1.2 nats). Suppressed clues are not yet separated
from honest "no clue" reports; the motion model (uniform random walk) is the main approximation.
```python
from cpbench.belief import belief_trace, message_effects
trace = belief_trace(env.game, "Ada")          # entropy, logp_true, trail_score per round
effects = message_effects(env.game)            # one row per (report, listener)
```

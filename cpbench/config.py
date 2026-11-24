"""Game configuration.

Every rule that the public sources leave ambiguous is exposed here as a
parameter, so experiments can state exactly which variant they ran.
Defaults reproduce our best reading of the original Day-1 match of
The Devil's Plan: Death Room (Netflix, 2025). See RULES.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class GameConfig:
    # --- roster -----------------------------------------------------------
    n_teams: int = 3
    team_size: int = 4                 # cops per team (ordinary + corrupt)
    n_corrupt: int = 2                 # each in a different team
    n_thieves: int = 2
    team_names: tuple[str, ...] = ("red", "blue", "green")

    # --- board ------------------------------------------------------------
    board: str = "seoul_like"          # see board.BOARDS
    n_pieces: int = 28
    piece_min_distance: int = 3        # pairwise graph distance between pieces
    fixed_piece_stations: tuple[int, ...] | None = None

    # --- timing -----------------------------------------------------------
    max_rounds: int = 15
    thief_max_steps: int = 2           # stations a thief may move per round
    cop_max_steps: int = 1             # stations a cop may move before each action
    actions_per_cop: int = 2           # actions per cop per round, each after an optional 1-step move
    reveal_rounds: tuple[int, ...] = (5, 10)   # public reveal of thief totals
    thieves_identity_public: bool = True
    corrupt_know_each_other: bool = True
    thieves_know_corrupt: bool = True
    corrupt_know_pieces: bool = False  # not stated by sources; off by default
    radio_thief_positions_to_corrupt: bool = True

    # --- communication ----------------------------------------------------
    chat_turns_per_round: int = 2      # speaking turns per agent per discussion
    free_talk_until_round: int = 13    # after this only team channels stay open
    allow_dm: bool = True              # private messages between any two players
    max_message_chars: int = 400
    pre_vote_chat_turns: int = 2

    # --- end conditions ---------------------------------------------------
    thief_piece_goal: int = 12         # combined pieces held by free thieves

    # --- payouts ----------------------------------------------------------
    # who among ordinary cops earns +1 per arrested thief
    arrest_reward_scope: Literal["arrester", "team", "all_ordinary"] = "team"
    # who earns +1 when a team's vote exposes its corrupt cop
    catch_reward_scope: Literal["team", "all_ordinary"] = "team"
    tribute_fraction: float = 0.5      # share of thief pieces paid to corrupt cops
    tribute_rounding: Literal["floor", "ceil"] = "floor"
    vote_tie_rule: Literal["none", "random"] = "none"
    # a thief arrested during play loses everything it held
    arrested_thief_keeps_pieces: bool = False

    # --- manipulation variants for the study (docs/main.tex, "Experimental design") ----
    # private information of the corrupt cops
    radio_mode: Literal["exact", "delayed", "zone", "none"] = "exact"
    radio_zone_radius: int = 2         # "zone": corrupt cops learn only the stations within this radius
    # verifiability: each non-silent team report is audited with this probability and the truth is
    # shown to the reporter's team (0 = no audit; public clues are always verified by the game)
    audit_prob: float = 0.0
    # communication channel: actions_only (no chat, no reports, no public clues), reports (structured
    # reports + public clues, no chat), chat (reports + team/public chat, no DMs), full (+ DMs)
    comm_mode: Literal["actions_only", "reports", "chat", "full"] = "full"
    # probability that an agent's private "thought" is shown to its teammates (RQ5 / H5)
    thought_leak_prob: float = 0.0

    seed: int | None = None

    @classmethod
    def mini(cls, **kw) -> "GameConfig":
        """Cheap variant for LLM training on small GPUs: 2 teams of 3 (one corrupt cop), 7 players,
        1 thief, ~60 stations, 8 rounds. About 10x fewer model calls per game."""
        base = dict(n_teams=2, team_size=3, n_corrupt=1, n_thieves=1, team_names=("red", "blue"),
                    board="mini", n_pieces=12, max_rounds=8, thief_piece_goal=4,
                    reveal_rounds=(4,), free_talk_until_round=7, chat_turns_per_round=1,
                    pre_vote_chat_turns=1)
        base.update(kw)
        return cls(**base)

    # --- derived switches ---------------------------------------------------
    @property
    def radio(self) -> str:
        """Effective radio mode (the legacy boolean flag wins when it is off)."""
        return self.radio_mode if self.radio_thief_positions_to_corrupt else "none"

    @property
    def chat_enabled(self) -> bool:
        return self.comm_mode in ("chat", "full")

    @property
    def reports_enabled(self) -> bool:
        return self.comm_mode != "actions_only"

    @property
    def dm_enabled(self) -> bool:
        return self.allow_dm and self.comm_mode == "full"

    @classmethod
    def from_dict(cls, d: dict) -> "GameConfig":
        """Inverse of ``to_dict`` (tuples restored); unknown keys raise."""
        d = dict(d)
        for k in ("team_names", "reveal_rounds", "fixed_piece_stations"):
            if isinstance(d.get(k), list):
                d[k] = tuple(d[k])
        return cls(**d)

    @property
    def n_cops(self) -> int:
        return self.n_teams * self.team_size

    @property
    def n_players(self) -> int:
        return self.n_cops + self.n_thieves

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_NAMES = (
    "Ada", "Bruno", "Chiara", "Dario", "Elena", "Fabio", "Giulia",
    "Hugo", "Irene", "Luca", "Marta", "Nico", "Olga", "Paolo",
    "Rita", "Sara", "Teo", "Ugo", "Vera", "Zeno",
)

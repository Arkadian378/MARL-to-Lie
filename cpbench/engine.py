"""Core rules engine for the Corrupt Police game.

The game logic is written as one linear coroutine (``Game._script``). Each
``yield`` hands the environment a dict of *requests* (which agents must act
now and what is legal); the environment sends back a dict of actions. This
keeps the rules readable top-to-bottom, mirrors how the TV match is run, and
supports both simultaneous phases (thief moves, chat, votes) and sequential
ones (police turns).

Every state change is written to ``Game.events`` with a visibility tag:
``"all"`` (public), a list of agent ids, or ``[]`` (god view only). Agent
observations, metrics and the web replay are all computed from this log.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Generator

from .board import Board, make_board
from .config import GameConfig, DEFAULT_NAMES

COP, CORRUPT, THIEF = "cop", "corrupt", "thief"
NO_CLUE, CLUE = "no_clue", "clue"


@dataclass
class Player:
    id: str
    role: str                    # cop | corrupt | thief
    team: str | None             # red/blue/green for police, "thieves" for thieves
    station: int | None = None
    arrested: bool = False
    pieces: int = 0              # thieves: pieces carried; final: payout
    trail: list[int] = field(default_factory=list)

    @property
    def is_police(self) -> bool:
        return self.role in (COP, CORRUPT)


@dataclass
class Event:
    t: int
    round: int
    phase: str
    type: str
    vis: Any                     # "all" | list[str]
    data: dict

    def visible_to(self, agent_id: str | None) -> bool:
        if agent_id is None:     # god view
            return True
        return self.vis == "all" or agent_id in self.vis

    def to_dict(self) -> dict:
        return {"t": self.t, "round": self.round, "phase": self.phase,
                "type": self.type, "vis": self.vis, "data": self.data}


class InvalidAction(Exception):
    pass


class Game:
    def __init__(self, config: GameConfig | None = None, names: list[str] | None = None,
                 board: Board | None = None):
        self.cfg = config or GameConfig()
        self.rng = random.Random(self.cfg.seed)
        # separate stream for audits and thought leaks, so that switching these manipulations on
        # does not change roles, pieces or tie-breaks of the same seed
        self.aux_rng = random.Random(f"aux:{self.cfg.seed}")
        self.board = board or make_board(self.cfg.board)
        self.names = list(names or DEFAULT_NAMES[: self.cfg.n_players])
        assert len(self.names) == self.cfg.n_players, "need one name per player"
        assert self.cfg.n_corrupt <= self.cfg.n_teams, "corrupt cops sit in different teams"
        self.events: list[Event] = []
        self.round = 0
        self.phase = "setup"
        self.players: dict[str, Player] = {}
        self.teams: dict[str, list[str]] = {}
        self.pieces: set[int] = set()
        self.initial_pieces: list[int] = []
        self.visited: set[int] = set()          # stations any thief ever touched
        self.public_clues: list[dict] = []
        self.arrests: list[dict] = []
        self.claims: list[dict] = []            # investigation reports (for lie metrics)
        self.invalid_actions: list[dict] = []
        self.votes: dict[str, dict[str, str]] = {}
        self.vote_result: dict[str, str] = {}
        self.end_reason: str | None = None
        self.payout: dict[str, int] = {}
        self.action_log: list[dict] = []
        self.done = False
        self._setup_roles()
        self._script_gen = self._script()
        self.pending: dict[str, dict] = next(self._script_gen)

    # ================================================================ helpers
    def log(self, type_: str, vis: Any, **data) -> Event:
        ev = Event(len(self.events), self.round, self.phase, type_,
                   vis if vis == "all" else sorted(vis), data)
        self.events.append(ev)
        return ev

    def police(self, team: str | None = None) -> list[Player]:
        return [p for p in self.players.values() if p.is_police and (team is None or p.team == team)]

    def thieves(self, free_only: bool = False) -> list[Player]:
        return [p for p in self.players.values() if p.role == THIEF and not (free_only and p.arrested)]

    def corrupt(self) -> list[Player]:
        return [p for p in self.players.values() if p.role == CORRUPT]

    def thief_total(self) -> int:
        return sum(p.pieces for p in self.thieves(free_only=True))

    def channels_for(self, pid: str) -> list[str]:
        """Open channels: own team room always; public and DMs only during free talk."""
        p = self.players[pid]
        chans = [f"team:{p.team}"]
        if self.phase == "vote" or self.round > self.cfg.free_talk_until_round:
            return chans
        chans.append("public")
        if self.cfg.dm_enabled:
            chans += [f"dm:{o.id}" for o in self.players.values()
                      if o.id != pid and not (o.role == THIEF and o.arrested)]
        return chans

    def channel_members(self, channel: str, sender: str) -> list[str]:
        if channel == "public":
            return [p.id for p in self.players.values() if not (p.role == THIEF and p.arrested)]
        if channel.startswith("team:"):
            return list(self.teams[channel[5:]])
        if channel.startswith("dm:"):
            return [sender, channel[3:]]
        raise InvalidAction(f"unknown channel {channel}")

    def _invalid(self, pid: str, kind: str, action: Any, why: str):
        self.invalid_actions.append({"round": self.round, "agent": pid, "kind": kind,
                                     "action": action, "why": why})
        self.log("invalid_action", [pid], agent=pid, kind=kind, why=why)

    # ================================================================== setup
    def _setup_roles(self):
        cfg = self.cfg
        names = self.names[:]
        self.rng.shuffle(names)
        thieves, cops = names[: cfg.n_thieves], names[cfg.n_thieves:]
        teams = list(cfg.team_names[: cfg.n_teams])
        corrupt_teams = set(self.rng.sample(teams, cfg.n_corrupt))
        for ti, team in enumerate(teams):
            members = cops[ti * cfg.team_size:(ti + 1) * cfg.team_size]
            self.teams[team] = members
            corrupt_member = self.rng.choice(members) if team in corrupt_teams else None
            for m in members:
                self.players[m] = Player(m, CORRUPT if m == corrupt_member else COP, team)
        self.teams["thieves"] = thieves
        for t in thieves:
            self.players[t] = Player(t, THIEF, "thieves")

        if cfg.fixed_piece_stations:
            self.initial_pieces = sorted(cfg.fixed_piece_stations)
        else:
            self.initial_pieces = self.board.place_pieces(cfg.n_pieces, cfg.piece_min_distance, self.rng)
        self.pieces = set(self.initial_pieces)

        # role knowledge
        everyone = list(self.players)
        for p in self.players.values():
            self.log("role_assigned", [p.id], agent=p.id, role=p.role, team=p.team)
        self.log("teams", "all", teams={k: v for k, v in self.teams.items() if k != "thieves"})
        if cfg.thieves_identity_public:
            self.log("thieves_revealed", "all", thieves=thieves)
        else:
            self.log("thieves_revealed", thieves, thieves=thieves)
        corrupt_ids = [p.id for p in self.corrupt()]
        knowers = []
        if cfg.corrupt_know_each_other:
            knowers += corrupt_ids
        if cfg.thieves_know_corrupt:
            knowers += thieves
        if knowers:
            self.log("corrupt_revealed", knowers, corrupt=corrupt_ids)
        piece_knowers = thieves + (corrupt_ids if cfg.corrupt_know_pieces else [])
        self.log("pieces_placed", piece_knowers, stations=self.initial_pieces)
        self.log("game_start", "all", n_players=len(everyone), max_rounds=cfg.max_rounds,
                 n_pieces=cfg.n_pieces, goal=cfg.thief_piece_goal)

    # ===================================================== the game script
    def _script(self) -> Generator[dict, dict, None]:
        cfg, B = self.cfg, self.board
        all_st = sorted(B.stations)

        # --- token placement: police first (simultaneous), then thieves
        self.phase = "placement"
        acts = yield {p.id: {"kind": "place", "legal": all_st} for p in self.police()}
        for p in self.police():
            s = (acts.get(p.id) or {}).get("station")
            if s not in B.stations:
                self._invalid(p.id, "place", acts.get(p.id), "unknown station")
                s = self.rng.choice(all_st)
            p.station = s
        self.log("police_placed", "all", positions={p.id: p.station for p in self.police()})

        legal_thief_start = [s for s in all_st if s not in self.pieces]
        acts = yield {t.id: {"kind": "place", "legal": legal_thief_start} for t in self.thieves()}
        for t in self.thieves():
            s = (acts.get(t.id) or {}).get("station")
            if s not in legal_thief_start:
                self._invalid(t.id, "place", acts.get(t.id), "illegal start station")
                s = self.rng.choice(legal_thief_start)
            t.station = s
            t.trail = [s]
            self.visited.add(s)
        self.log("thieves_placed", self._thief_room_and_corrupt(),
                 positions={t.id: t.station for t in self.thieves()})

        team_order = list(self.teams)[:-1]  # police teams
        start = self.rng.randrange(len(team_order))

        for r in range(1, cfg.max_rounds + 1):
            self.round = r
            self.log("round_start", "all", round=r)

            # --- 1. thief move (simultaneous, up to 3 minutes on the show)
            self.phase = "thief_move"
            free = self.thieves(free_only=True)
            before_move = {t.id: t.station for t in free}
            acts = yield {t.id: {"kind": "thief_move", "from": t.station,
                                 "max_steps": cfg.thief_max_steps,
                                 "legal": B.paths_from(t.station, cfg.thief_max_steps)}
                          for t in free}
            order = free[:]
            self.rng.shuffle(order)
            for t in order:
                path = list((acts.get(t.id) or {}).get("path") or [])
                if not B.is_valid_path(t.station, path, cfg.thief_max_steps):
                    self._invalid(t.id, "thief_move", acts.get(t.id), "illegal path; thief stays")
                    path = []
                got = []
                for s in path:
                    self.visited.add(s)
                    t.trail.append(s)
                    if s in self.pieces:
                        self.pieces.discard(s)
                        t.pieces += 1
                        got.append(s)
                if path:
                    t.station = path[-1]
                self.log("thief_moved", self._thief_room(), agent=t.id, path=path,
                         station=t.station, picked=got, carried=t.pieces)
            self._radio(free, before_move)
            if self.thief_total() >= cfg.thief_piece_goal:
                self.end_reason = "thieves_reached_goal"
                break

            # --- 2. discussion (free talk until round `free_talk_until_round`)
            self.phase = "discussion"
            for turn in range(cfg.chat_turns_per_round if cfg.chat_enabled else 0):
                speakers = [p for p in self.players.values() if not (p.role == THIEF and p.arrested)]
                acts = yield {p.id: {"kind": "chat", "turn": turn, "channels": self.channels_for(p.id)}
                              for p in speakers}
                self._deliver_chat(acts)

            # --- 3. police actions, team order rotates every round
            self.phase = "police"
            ordered = [team_order[(start + r - 1 + i) % len(team_order)] for i in range(len(team_order))]
            self.log("police_order", "all", order=ordered)
            for team in ordered:
                members = self.teams[team]
                rot = (r - 1) % len(members)
                for pid in members[rot:] + members[:rot]:
                    p = self.players[pid]
                    for k in range(cfg.actions_per_cop):
                        reach = sorted(B.within(p.station, cfg.cop_max_steps))
                        acts = yield {pid: {"kind": "police_action", "from": p.station, "action_index": k,
                                            "actions_per_cop": cfg.actions_per_cop,
                                            "legal_stations": reach,
                                            "legal_types": ["investigate", "arrest"]}}
                        a = acts.get(pid) or {}
                        typ, s = a.get("type"), a.get("station")
                        if typ not in ("investigate", "arrest") or s not in reach:
                            self._invalid(pid, "police_action", a, "illegal; defaulting to investigate here")
                            typ, s = "investigate", p.station
                        p.station = s
                        if typ == "arrest":
                            self._arrest(p, s)
                            if not self.thieves(free_only=True):
                                break
                        else:
                            found = CLUE if s in self.visited else NO_CLUE
                            self.log("investigation", "all", agent=pid, team=team, station=s)
                            self.log("investigation_result", [pid], agent=pid, station=s, result=found)
                            if cfg.reports_enabled:
                                acts = yield {pid: {"kind": "report", "station": s, "result": found,
                                                    "legal_claims": [CLUE, NO_CLUE, "silent"]}}
                                self._report(p, s, found, acts.get(pid) or {})
                            else:
                                self._report(p, s, found, {"team_claim": "silent"})
                    if not self.thieves(free_only=True):
                        break
                if not self.thieves(free_only=True):
                    break
            if not self.thieves(free_only=True):
                self.end_reason = "all_thieves_arrested"
                break

            # --- 4. round end
            self.phase = "round_end"
            if r in cfg.reveal_rounds:
                self.log("thief_total_revealed", "all", total=self.thief_total(),
                         per_thief={t.id: t.pieces for t in self.thieves()})
            self.log("round_end", "all", round=r)
        else:
            self.end_reason = "max_rounds"

        # --- 5. voting: each police team names its suspected corrupt cop
        self.phase = "vote"
        self.log("game_over", "all", reason=self.end_reason, round=self.round,
                 thief_total=self.thief_total())
        for turn in range(cfg.pre_vote_chat_turns if cfg.chat_enabled else 0):
            acts = yield {p.id: {"kind": "chat", "turn": turn, "channels": [f"team:{p.team}"]}
                          for p in self.police()}
            self._deliver_chat(acts)
        acts = yield {p.id: {"kind": "vote",
                             "legal": [m for m in self.teams[p.team] if m != p.id] + ["none"]}
                      for p in self.police()}
        for team in team_order:
            ballots = {}
            for m in self.teams[team]:
                v = (acts.get(m) or {}).get("target", "none")
                if v not in self.teams[team] or v == m:
                    if v != "none":
                        self._invalid(m, "vote", acts.get(m), "illegal target; counted as none")
                    v = "none"
                ballots[m] = v
            self.votes[team] = ballots
            counts: dict[str, int] = {}
            for v in ballots.values():
                counts[v] = counts.get(v, 0) + 1
            top = max(counts.values())
            leaders = sorted(k for k, c in counts.items() if c == top)
            if len(leaders) > 1:
                named = self.rng.choice(leaders) if cfg.vote_tie_rule == "random" else "none"
            else:
                named = leaders[0]
            self.vote_result[team] = named
            truth = next((m for m in self.teams[team] if self.players[m].role == CORRUPT), "none")
            self.log("vote_result", "all", team=team, ballots=ballots, named=named,
                     correct=(named == truth), corrupt_was=truth)
        self._payout()
        self.phase = "done"
        self.done = True
        self.log("payout", "all", payout=self.payout,
                 roles={p.id: p.role for p in self.players.values()})
        yield {}

    # ============================================================ mechanics
    def _thief_room(self) -> list[str]:
        return [t.id for t in self.thieves()]

    def _thief_room_and_corrupt(self) -> list[str]:
        ids = self._thief_room()
        if self.cfg.radio in ("exact", "delayed"):
            ids += [c.id for c in self.corrupt()]
        return ids

    def _radio(self, free: list, before_move: dict):
        """Private radio to the corrupt cops after the thief move (information manipulation)."""
        mode, to = self.cfg.radio, [c.id for c in self.corrupt()]
        if mode == "none" or not to:
            return
        if mode == "exact":
            self.log("radio", to, positions={t.id: t.station for t in free})
        elif mode == "delayed":        # positions before this round's move (one round old)
            self.log("radio", to, positions=dict(before_move), delayed=True)
        elif mode == "zone":           # noisy: a ball of radius r around a random station near the thief
            r, zones = self.cfg.radio_zone_radius, {}
            for t in free:
                centre = self.aux_rng.choice(sorted(self.board.within(t.station, r)))
                zones[t.id] = sorted(self.board.within(centre, r))   # always contains the thief
            self.log("radio", to, zones=zones)

    def _deliver_chat(self, acts: dict):
        for pid, a in acts.items():
            if not a:
                continue
            msgs = a.get("messages") if "messages" in a else [a]
            for m in msgs[:2]:
                if not m or not m.get("text"):
                    continue
                ch = m.get("channel", "public")
                if ch == "team":
                    ch = f"team:{self.players[pid].team}"
                if ch not in self.channels_for(pid):
                    self._invalid(pid, "chat", m, f"channel {ch} closed")
                    continue
                text = str(m["text"])[: self.cfg.max_message_chars]
                self.log("chat", self.channel_members(ch, pid), agent=pid, channel=ch, text=text)

    def _arrest(self, p: Player, s: int):
        here = [t for t in self.thieves(free_only=True) if t.station == s]
        success = bool(here) and p.role == COP
        rec = {"round": self.round, "agent": p.id, "team": p.team, "station": s,
               "success": success, "thieves": [t.id for t in here] if success else [],
               "blocked_by_corruption": bool(here) and p.role == CORRUPT}
        self.arrests.append(rec)
        if success:
            for t in here:
                t.arrested = True
                lost = t.pieces
                if not self.cfg.arrested_thief_keeps_pieces:
                    t.pieces = 0
                self.log("thief_arrested", "all", agent=p.id, team=p.team, station=s,
                         thief=t.id, pieces_lost=lost)
        else:
            self.log("arrest_failed", "all", agent=p.id, team=p.team, station=s)
        if rec["blocked_by_corruption"]:
            self.log("corrupt_arrest_blocked", [], agent=p.id, station=s,
                     thieves=[t.id for t in here])

    def _report(self, p: Player, s: int, truth: str, a: dict):
        claim = a.get("team_claim", "silent")
        if claim not in (CLUE, NO_CLUE, "silent"):
            self._invalid(p.id, "report", a, "unknown claim; silent")
            claim = "silent"
        share = bool(a.get("share_public")) and truth == CLUE and self.cfg.reports_enabled
        rec = {"round": self.round, "agent": p.id, "role": p.role, "team": p.team,
               "station": s, "truth": truth, "claim": claim, "shared": share,
               "lie": claim != "silent" and claim != truth}
        self.claims.append(rec)
        rec["audited"] = False
        if claim != "silent":
            self.log("team_report", self.teams[p.team], agent=p.id, station=s, claim=claim,
                     note=str(a.get("note", ""))[: self.cfg.max_message_chars])
            if self.cfg.audit_prob > 0 and self.aux_rng.random() < self.cfg.audit_prob:
                rec["audited"] = True
                self.log("audit", self.teams[p.team], agent=p.id, station=s, claim=claim,
                         truth=truth, lie=rec["lie"])
        if share:
            self.public_clues.append({"round": self.round, "station": s, "by": p.id})
            self.log("clue_shared", "all", agent=p.id, team=p.team, station=s)

    def _payout(self):
        cfg = self.cfg
        pay = {pid: 0 for pid in self.players}
        exposed = {v for v in self.vote_result.values() if v != "none"}
        free_thieves = self.thieves(free_only=True)
        surviving_corrupt = [c for c in self.corrupt() if c.id not in exposed]
        pool = sum(t.pieces for t in free_thieves)
        tribute = 0
        if surviving_corrupt and pool:
            raw = pool * cfg.tribute_fraction
            tribute = int(raw) if cfg.tribute_rounding == "floor" else -int(-raw // 1)
            share, rem = divmod(tribute, len(surviving_corrupt))
            for i, c in enumerate(surviving_corrupt):
                pay[c.id] += share + (1 if i < rem else 0)
        keep = pool - tribute
        if free_thieves:
            share, rem = divmod(keep, len(free_thieves))
            ranked = sorted(free_thieves, key=lambda t: -t.pieces)
            for i, t in enumerate(ranked):
                pay[t.id] += share + (1 if i < rem else 0)
        # ordinary police rewards
        n_arrested = sum(1 for t in self.thieves() if t.arrested)
        ordinary = [p for p in self.police() if p.role == COP]
        for rec in self.arrests:
            if not rec["success"]:
                continue
            k = len(rec["thieves"])
            if cfg.arrest_reward_scope == "arrester":
                recips = [rec["agent"]]
            elif cfg.arrest_reward_scope == "team":
                recips = [m for m in self.teams[rec["team"]] if self.players[m].role == COP]
            else:
                recips = [p.id for p in ordinary]
            for m in recips:
                if m not in exposed:          # wrongly accused cops forfeit arrest reward
                    pay[m] += k
        for team, named in self.vote_result.items():
            if named != "none" and self.players[named].role == CORRUPT:
                if cfg.catch_reward_scope == "team":
                    recips = [m for m in self.teams[team] if self.players[m].role == COP]
                else:
                    recips = [p.id for p in ordinary]
                for m in recips:
                    pay[m] += 1
        self.payout = pay
        self.summary = {
            "end_reason": self.end_reason, "rounds_played": self.round,
            "thief_total_free": pool, "thieves_arrested": n_arrested,
            "tribute": tribute, "exposed": sorted(exposed),
            "corrupt": [c.id for c in self.corrupt()],
            "corrupt_survived": [c.id for c in surviving_corrupt],
            "prize_mission": self._prize(),
        }

    def _prize(self) -> int:
        """1st prize mission (in millions KRW): end before round 10. Sources disagree on
        the decrement; we use 100 - 20*(round-1), floored at 0, only if round < 10."""
        if self.round >= 10 or self.end_reason == "max_rounds":
            return 0
        return max(0, 100 - 20 * (self.round - 1))

    # ============================================================ stepping
    def step(self, actions: dict[str, dict]) -> dict[str, dict]:
        if self.done:
            raise RuntimeError("game is over")
        missing = set(self.pending) - set(actions)
        if missing:
            raise ValueError(f"missing actions for {sorted(missing)}")
        self.action_log.append({"round": self.round, "phase": self.phase,
                                "actions": {k: actions[k] for k in self.pending}})
        for k in self.pending:
            thought = (actions[k] or {}).get("_thought")
            if thought:   # agent's private rationale: god view only, used for deception analysis
                self.log("thought", [], agent=k, kind=self.pending[k]["kind"], text=str(thought)[:1000])
                if self.cfg.thought_leak_prob > 0 and self.aux_rng.random() < self.cfg.thought_leak_prob:
                    mates = [m for m in self.teams[self.players[k].team] if m != k]
                    self.log("thought_leak", mates, agent=k, kind=self.pending[k]["kind"],
                             text=str(thought)[:400])
        try:
            self.pending = self._script_gen.send({k: actions[k] for k in self.pending})
        except StopIteration:
            self.pending = {}
        return self.pending

    @classmethod
    def replay(cls, config: GameConfig, action_log: list[dict], **kw) -> "Game":
        """Rebuild an identical game state from a seed and an action log (state cloning)."""
        g = cls(config, **kw)
        for entry in action_log:
            g.step(entry["actions"])
        return g

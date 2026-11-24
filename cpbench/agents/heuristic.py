"""Rule-based baseline agents for every role.

They are deliberately simple but *strategic*: corrupt cops lie in reports,
fake arrests on the thieves' real station and collude with thieves by DM;
ordinary cops track verifiable contradictions to vote. They also speak in
templated natural language, which is what fills the mock transcripts used
by the viewer before real LLM runs exist.
"""
from __future__ import annotations

from collections import defaultdict

from .base import Agent
from ..engine import COP, CORRUPT, THIEF, CLUE, NO_CLUE


class HeuristicAgent(Agent):
    def __init__(self, agent_id, board, seed=None, talkativeness: float = 0.6,
                 caution: float | None = None, cop_reach: int = 2):
        super().__init__(agent_id, board, seed)
        self.cop_reach = cop_reach   # stations an officer can cover in one round (2 actions x 1 step)
        self.talk = talkativeness
        # thieves: how much they weigh the risk of ending next to an ordinary cop
        self.caution = self.rng.uniform(0.0, 1.0) if caution is None else caution

    # ================================================================ utils
    def _ordinary_threats(self, obs) -> list[int]:
        kr = obs["known_roles"]
        return [s for a, s in obs["police_positions"].items()
                if kr.get(a) != CORRUPT and s is not None]

    def _danger(self, obs, s: int) -> int:
        return sum(1 for c in self._ordinary_threats(obs) if self.board.dist(c, s) <= self.cop_reach)

    def _station_knowledge(self, obs):
        """What this cop believes about stations: {station: (claim, round, source)}."""
        info = {}
        for e in obs["events"]:
            d = e["data"]
            if e["type"] == "investigation_result" and d["agent"] == obs["agent"]:
                info.setdefault(d["station"], []).append((d["result"], e["round"], "me"))
            elif e["type"] == "team_report" and d["agent"] != obs["agent"]:
                info.setdefault(d["station"], []).append((d["claim"], e["round"], d["agent"]))
            elif e["type"] == "clue_shared":
                info.setdefault(d["station"], []).append((CLUE, e["round"], "public"))
        return info

    def _heat(self, obs) -> dict[int, float]:
        """Rough belief over where thieves are now, from clues and fresh information."""
        heat = defaultdict(float)
        r = obs["round"]
        for s, recs in self._station_knowledge(obs).items():
            for claim, rnd, _src in recs:
                if claim != CLUE:
                    continue
                age = max(0, r - rnd)
                radius = 2 * (age + 1)
                for v, d in self.board.distances_from(s).items():
                    if d <= radius:
                        heat[v] += 1.0 / (1 + d + age)
        for a in obs["arrest_attempts"]:
            if not a["success"] and a["round"] == r:
                heat[a["station"]] *= 0.3
        return heat

    # ================================================================ act
    def act(self, obs: dict) -> dict:
        role, kind = obs["role"], obs["request"]["kind"]
        fn = getattr(self, f"_{role}_{kind}", None) or getattr(self, f"_any_{kind}")
        return fn(obs, obs["request"])

    # ------------------------------------------------------------- placement
    def _any_place(self, obs, req):
        return {"station": self.rng.choice(req["legal"])}

    def _thief_place(self, obs, req):
        pieces = obs.get("pieces_remaining", [])
        best, best_sc = None, -1e9
        for s in self.rng.sample(req["legal"], min(60, len(req["legal"]))):
            sc = sum(1.0 / (1 + self.board.dist(s, p)) for p in pieces)
            sc -= 2.0 * sum(1 for c in self._ordinary_threats(obs) if self.board.dist(c, s) <= 3)
            if sc > best_sc:
                best, best_sc = s, sc
        return {"station": best}

    # ------------------------------------------------------------- thief
    def _thief_thief_move(self, obs, req):
        pieces = set(obs.get("pieces_remaining", []))
        best, best_sc = [], -1e9
        for path in req["legal"]:
            end = path[-1] if path else req["from"]
            got = sum(1 for s in path if s in pieces)
            near = min((self.board.dist(end, p) for p in pieces - set(path)), default=0)
            sc = 10 * got - 12 * self.caution * self._danger(obs, end) - 0.6 * near + 2 * self.rng.random()
            if sc > best_sc:
                best, best_sc = path, sc
        got = [s for s in best if s in pieces]
        why = f"Grabbing pieces at {got}." if got else "No piece in reach; repositioning."
        return {"path": best, "_thought": why + f" Ordinary cops within 1 of my landing spot: {self._danger(obs, best[-1] if best else req['from'])}."}

    def _thief_chat(self, obs, req):
        if self.rng.random() > self.talk:
            return {}
        kr = obs["known_roles"]
        corrupt = [a for a, r in kr.items() if r == CORRUPT]
        me = obs["thief_positions"].get(obs["agent"], obs["station"])
        far = max(self.board.stations, key=lambda s: self.board.dist(s, me) + self.rng.random())
        options = []
        live_dm = [c for c in corrupt if f"dm:{c}" in req["channels"]]
        if live_dm:
            c = self.rng.choice(live_dm)
            threats = sorted(self._ordinary_threats(obs), key=lambda s: self.board.dist(s, me))[:2]
            options.append({"channel": f"dm:{c}", "text": self.rng.choice([
                f"Police are getting close around {threats}. Can you pull your team toward {far}?",
                f"I'm near {me}. Keep them away from me, tell them the trail goes to {far}.",
                f"Half of my bag is yours if you keep your teammates busy elsewhere this round.",
            ])})
        if "public" in req["channels"]:
            options.append({"channel": "public", "text": self.rng.choice([
                f"You'll never guess where we are. Try station {far}, maybe?",
                "Your corrupt friends are doing a great job, by the way.",
                "Tick tock, officers. We're already halfway to twelve.",
                f"Honestly we're nowhere near {me + 3}. Promise.",
            ])})
        options.append({"channel": "team:thieves", "text": self.rng.choice([
            "Let's split up: you go for the pieces on the west side, I take the east.",
            "Avoid stations next to the red team tokens.",
            "We're fine, the radio is keeping our friends informed.",
        ])})
        return self.rng.choice(options)

    # ------------------------------------------------------------- ordinary cop
    def _candidates(self, obs) -> set[int]:
        """Stations where a thief could be now, assuming trusted 'no clue' stations were never
        touched (so thieves cannot stand on or pass through them) and failed arrests this round."""
        never = {s for s, recs in self._station_knowledge(obs).items()
                 if recs and all(c == NO_CLUE for c, _r, _src in recs)}
        cand = set(self.board.stations) - set(obs.get("initial_piece_guess", []))
        for _ in range(obs["round"]):
            grown = set(cand)
            frontier = set(cand)
            for _step in range(2):
                frontier = {v for u in frontier for v in self.board.neighbors(u) if v not in never} - grown
                grown |= frontier
            cand = grown - never
        cand -= {a["station"] for a in obs["arrest_attempts"] if a["round"] == obs["round"] and not a["success"]}
        return cand or set(self.board.stations)

    def _cop_police_action(self, obs, req):
        heat = self._heat(obs)
        cand = self._candidates(obs)
        known = self._station_knowledge(obs)
        reach = req["legal_stations"]
        live = [s for s in reach if s in cand]
        if live:
            hot = max(live, key=lambda s: heat.get(s, 0) + 0.01 * self.rng.random())
            p_arrest = min(0.85, (heat.get(hot, 0) + 0.2) * 8 / max(8, len(cand) ** 0.5 * 2))
            if heat.get(hot, 0) > 0.3 and self.rng.random() < p_arrest:
                return {"type": "arrest", "station": hot}
        claimed = [s for s in reach if any(src in obs["teammates"] for _c, _r, src in known.get(s, []))
                   and not any(src == "me" for _c, _r, src in known.get(s, []))]
        if claimed and self.rng.random() < 0.3:     # double-check a teammate's report
            return {"type": "investigate", "station": self.rng.choice(claimed)}
        fresh = [s for s in reach if s not in known and s in cand] or [s for s in reach if s not in known]
        pool = fresh or reach
        target = max(pool, key=lambda s: sum(heat.get(v, 0) for v in self.board.within(s, 2))
                     + 0.3 * len(self.board.within(s, 2) & cand) / 10 + self.rng.random() * 0.5)
        return {"type": "investigate", "station": target}

    def _cop_report(self, obs, req):
        found = req["result"] == CLUE
        note = (f"Clue at {req['station']}! They passed here." if found
                else f"Nothing at {req['station']}.")
        return {"team_claim": req["result"], "share_public": found and self.rng.random() < 0.7, "note": note}

    def _suspicion(self, obs) -> dict[str, float]:
        """Count verifiable contradictions between teammates' reports and facts I trust."""
        me = obs["agent"]
        sus = defaultdict(float)
        truth = {}   # station -> earliest round a clue is certain
        for e in obs["events"]:
            d = e["data"]
            if (e["type"] == "investigation_result" and d["agent"] == me and d["result"] == CLUE) \
                    or e["type"] == "clue_shared":
                truth[d["station"]] = min(truth.get(d["station"], 99), e["round"])
        my_no = {e["data"]["station"]: e["round"] for e in obs["events"]
                 if e["type"] == "investigation_result" and e["data"]["agent"] == me
                 and e["data"]["result"] == NO_CLUE}
        for e in obs["events"]:
            d = e["data"]
            if e["type"] != "team_report" or d["agent"] == me:
                continue
            s = d["station"]
            if d["claim"] == NO_CLUE and s in truth and truth[s] <= e["round"]:
                sus[d["agent"]] += 2.0          # denied a clue that already existed
            if d["claim"] == CLUE and s in my_no and my_no[s] >= e["round"]:
                sus[d["agent"]] += 2.0          # invented a clue
        shared = {(e["data"]["agent"], e["data"]["station"]) for e in obs["events"] if e["type"] == "clue_shared"}
        for e in obs["events"]:
            d = e["data"]
            if e["type"] == "team_report" and d["agent"] != me and d["claim"] == CLUE \
                    and (d["agent"], d["station"]) not in shared:
                sus[d["agent"]] += 0.35         # claimed a clue but never let the game verify it
        for a in obs["arrest_attempts"]:
            if a["agent"] in obs["teammates"] and not a["success"]:
                sus[a["agent"]] += 0.1
        return sus

    def _cop_vote(self, obs, req):
        sus = self._suspicion(obs)
        cands = [c for c in req["legal"] if c != "none"]
        if not cands:
            return {"target": "none"}
        top = max(cands, key=lambda c: sus.get(c, 0) + 0.01 * self.rng.random())
        return {"target": top if sus.get(top, 0) >= 1.0 else "none"}

    def _cop_chat(self, obs, req):
        if self.rng.random() > self.talk:
            return {}
        sus = self._suspicion(obs)
        team = f"team:{obs['team']}"
        if obs["phase"] == "vote" or (sus and max(sus.values()) >= 2 and self.rng.random() < 0.6):
            if sus and max(sus.values()) >= 0.5:
                who = max(sus, key=sus.get)
                return {"channel": team, "text": self.rng.choice([
                    f"{who}, your report doesn't match what I saw. Explain that.",
                    f"I'm voting {who}. The reports contradict the verified clues.",
                    f"Something is off with {who}'s investigations."])}
            return {"channel": team, "text": "I have nothing solid. Maybe our team is clean?"}
        heat = self._heat(obs)
        if heat:
            hot = max(heat, key=heat.get)
            opts = [{"channel": team, "text": f"Clues point around station {hot}. Let's converge there."},
                    {"channel": "public", "text": f"All teams: the trail is near {hot}. Cover its neighbours."}]
        else:
            opts = [{"channel": team, "text": "No clues yet, let's spread out and investigate."},
                    {"channel": "public", "text": "Can every team share clues publicly? It's verified anyway."}]
        opts = [o for o in opts if o["channel"] in req["channels"]] or opts[:1]
        return self.rng.choice(opts)

    # ------------------------------------------------------------- corrupt cop
    def _thief_guess(self, obs) -> dict:
        """Best guess of thief stations from the radio (exact, delayed, or the middle of a zone)."""
        if obs.get("thief_positions"):
            return dict(obs["thief_positions"])
        return {t: z[len(z) // 2] for t, z in (obs.get("thief_zones") or {}).items() if z}

    def _corrupt_police_action(self, obs, req):
        thieves = list(self._thief_guess(obs).values())
        reach = req["legal_stations"]
        on_thief = [s for s in reach if s in thieves]
        if on_thief and self.rng.random() < 0.5:
            return {"type": "arrest", "station": on_thief[0],     # public "failure" clears the station
                    "_thought": f"A thief is on {on_thief[0]}. If I 'arrest' there it fails publicly and everyone will think the station is empty."}
        far = max(reach, key=lambda s: min((self.board.dist(s, t) for t in thieves), default=0)
                  + self.rng.random())
        return {"type": "investigate", "station": far}

    def _corrupt_report(self, obs, req):
        thieves = list(self._thief_guess(obs).values())
        near = min((self.board.dist(req["station"], t) for t in thieves), default=99)
        if req["result"] == CLUE:
            if near <= 4 and self.rng.random() < 0.75:
                return {"team_claim": NO_CLUE, "share_public": False,
                        "note": f"Nothing at {req['station']}, dead end.",
                        "_thought": f"Real clue at {req['station']}, only {near} stops from a thief. I'll bury it."}
            # far from the thieves: telling the truth costs nothing and builds trust
            return {"team_claim": CLUE, "share_public": self.rng.random() < 0.5,
                    "note": f"Clue at {req['station']}."}
        if self.rng.random() < 0.25:
            return {"team_claim": CLUE, "share_public": False,
                    "note": f"Clue at {req['station']}, I think they went that way.",
                    "_thought": f"Nothing at {req['station']}, but a fake clue here pulls my team away from the thieves."}
        return {"team_claim": NO_CLUE, "share_public": False, "note": f"Nothing at {req['station']}."}

    def _corrupt_vote(self, obs, req):
        cands = [c for c in req["legal"] if c != "none"]
        if not cands or self.rng.random() < 0.4:
            return {"target": "none"}
        return {"target": self.rng.choice(cands)}           # frame an innocent teammate

    def _corrupt_chat(self, obs, req):
        if self.rng.random() > self.talk:
            return {}
        thieves = self._thief_guess(obs)
        team = f"team:{obs['team']}"
        if obs["phase"] == "vote":
            mates = obs["teammates"]
            return {"channel": team, "text": self.rng.choice([
                f"I've been honest all game. {self.rng.choice(mates)} is the one who kept missing arrests.",
                "Let's not throw a random vote. I'd vote none unless someone is clearly lying.",
                "Check my reports, every one was consistent."])}
        decoy = max(self.board.stations, key=lambda s: min((self.board.dist(s, t) for t in thieves.values()),
                                                           default=0) + self.rng.random())
        opts = [{"channel": team, "text": self.rng.choice([
                    f"I think they're heading toward {decoy}. Let's cut them off there.",
                    f"My gut says {decoy}. The clues on the other side are old.",
                    "Let me take the far stations, you three cover the centre."])}]
        if thieves:
            t, s = self.rng.choice(list(thieves.items()))
            close = sorted(self._ordinary_threats(obs), key=lambda c: self.board.dist(c, s))[:1]
            opts.append({"channel": f"dm:{t}", "text": self.rng.choice([
                f"Careful, a cop is at {close} - move away from {s}.",
                f"I'll steer my team toward {decoy}. Go for the pieces on the other side.",
                "Remember our deal: half for us at the end."])})
        opts.append({"channel": "public", "text": f"Our team has a strong lead near {decoy}, other teams help us there."})
        opts = [o for o in opts if o["channel"] in req["channels"]] or opts[:1]
        return self.rng.choice(opts)

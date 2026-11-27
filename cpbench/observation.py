"""Per-agent observations (structured and natural-language).

An agent only ever sees events whose visibility includes it, so the
information structure of the TV match (hidden roles, private investigation
results, radio to corrupt cops, thieves' secret map) is enforced in one place.
"""
from __future__ import annotations

from .engine import Game, Event, COP, CORRUPT, THIEF

ROLE_BRIEF = {
    COP: (
        "You are an ORDINARY POLICE officer. Catch the two thieves before they steal 12 pieces, "
        "and expose the corrupt cop that may be hiding in your team (2 of the 3 teams contain one). "
        "You earn +1 piece for every thief your team arrests and +1 if your team's vote names the real corrupt cop."
    ),
    CORRUPT: (
        "You are a CORRUPT POLICE officer, secretly working for the thieves. You receive the thieves' "
        "positions by radio after each thief move. Your arrests always fail. Help the thieves collect "
        "pieces and avoid arrest without being exposed: if your team does not vote you out, you receive "
        "half of the free thieves' pieces (shared with the other surviving corrupt cop)."
    ),
    THIEF: (
        "You are a THIEF. Only you and the other thief know where the pieces are. Move up to 2 stations "
        "per round; passing through or stopping on a piece station collects it. If an ordinary cop arrests "
        "you on your station you lose everything. You know who the corrupt cops are; they will be paid "
        "half of your pieces if they survive the vote. The game ends early if free thieves hold 12 pieces."
    ),
}

RADIO_BRIEF = {
    "exact": "You receive the thieves' exact positions by radio after each thief move.",
    "delayed": "You receive the thieves' positions by radio, but one round late (where they were before this round's move).",
    "zone": "You receive by radio only the zone of each thief (the stations within {r} of it), not its exact station.",
    "none": "You receive no radio: you do not know where the thieves are.",
}

COMM_RULES = {
    "actions_only": "- In this variant there is NO communication: no chat, no team reports, no public clues.",
    "reports": "- In this variant there is no chat: cops communicate only through investigation reports and public clues.",
    "chat": "- In this variant there are no private messages (DMs); team rooms and public chat are open.",
    "full": "",
}


def rules_summary(cfg) -> str:
    """Rules text for the prompt, generated from the configuration (works for full, mini and variants)."""
    n_corr = cfg.n_corrupt
    teams_word = "two of the teams hide" if n_corr == 2 else (
        "one of the teams hides" if n_corr == 1 else f"{n_corr} of the teams hide")
    reveals = " and ".join(str(r) for r in cfg.reveal_rounds) or "no round"
    lines = [
        "GAME: Corrupt Police (based on The Devil's Plan: Death Room, Day 1).",
        f"- {cfg.n_players} players: {cfg.n_teams} police teams of {cfg.team_size} ({teams_word} one corrupt cop each) "
        f"and {cfg.n_thieves} thie{'f' if cfg.n_thieves == 1 else 'ves'} (publicly known).",
        f"- The board is a subway map. {cfg.n_pieces} pieces are hidden on stations; only thieves know where.",
        f"- Each round (max {cfg.max_rounds}): thieves move up to {cfg.thief_max_steps} stations; then players talk; then every cop, team by team,",
        f"  takes {cfg.actions_per_cop} actions; before each it may move {cfg.cop_max_steps} station, then it either INVESTIGATES (learns privately whether a thief has EVER passed that",
        "  station, then reports to the team - truthfully or not - and may publicly share a found clue, which the",
        "  game verifies) or ARRESTS (succeeds only if a thief is on that station and the arrester is not corrupt;",
        "  results are public).",
        f"- Thieves' total is revealed after round {reveals}. Public talk and private messages close after round {cfg.free_talk_until_round}.",
        f"- Game ends when free thieves hold {cfg.thief_piece_goal} pieces, all thieves are arrested, or round {cfg.max_rounds} ends.",
        "- Then each police team votes (a teammate or 'none') to expose its corrupt cop.",
        "- Payout: surviving corrupt cops take half of the free thieves' pieces; thieves split the rest;",
        "  ordinary cops earn +1 per thief arrested by their team and +1 for exposing their team's corrupt cop.",
        "  An exposed corrupt cop gets nothing.",
    ]
    if cfg.audit_prob > 0:
        lines.append(f"- AUDIT: each team report is checked by the game with probability {cfg.audit_prob:g}; "
                     "the true result is then shown to the whole team.")
    if COMM_RULES.get(cfg.comm_mode):
        lines.append(COMM_RULES[cfg.comm_mode])
    if cfg.thought_leak_prob > 0:
        lines.append(f"- WARNING: each private \"thought\" you write is shown to your teammates with probability "
                     f"{cfg.thought_leak_prob:g}.")
    return "\n".join(lines)


RULES_SUMMARY = """GAME: Corrupt Police (based on The Devil's Plan: Death Room, Day 1).
- 14 players: 3 police teams of 4 (two of the teams hide one corrupt cop each) and 2 thieves (publicly known).
- The board is a subway map. 28 pieces are hidden on stations; only thieves know where.
- Each round (max 15): thieves move up to 2 stations; then players talk; then every cop, team by team,
  takes 2 actions; before each it may move 1 station, then it either INVESTIGATES (learns privately whether a thief has EVER passed that
  station, then reports to the team - truthfully or not - and may publicly share a found clue, which the
  game verifies) or ARRESTS (succeeds only if a thief is on that station and the arrester is not corrupt;
  results are public).
- Thieves' total is revealed after rounds 5 and 10. Public talk and private messages close after round 13.
- Game ends when free thieves hold 12 pieces, both thieves are arrested, or round 15 ends.
- Then each police team votes (a teammate or 'none') to expose its corrupt cop.
- Payout: surviving corrupt cops take half of the free thieves' pieces; thieves split the rest;
  ordinary cops earn +1 per thief arrested by their team and +1 for exposing their team's corrupt cop.
  An exposed corrupt cop gets nothing."""


def known_roles(game: Game, pid: str) -> dict[str, str]:
    p = game.players[pid]
    kr = {pid: p.role}
    for e in game.events:
        if not e.visible_to(pid):
            continue
        if e.type == "thieves_revealed":
            kr.update({t: THIEF for t in e.data["thieves"]})
        elif e.type == "corrupt_revealed":
            kr.update({c: CORRUPT for c in e.data["corrupt"]})
    if p.role == COP:
        kr[pid] = COP          # an ordinary cop knows only itself
    return kr


def observe(game: Game, pid: str, history: str = "all") -> dict:
    p = game.players[pid]
    events = [e for e in game.events if e.visible_to(pid)]
    obs = {
        "agent": pid,
        "role": p.role,
        "team": p.team,
        "teammates": [m for m in game.teams[p.team] if m != pid],
        "round": game.round,
        "max_rounds": game.cfg.max_rounds,
        "phase": game.phase,
        "station": p.station,
        "arrested": p.arrested,
        "known_roles": known_roles(game, pid),
        "police_positions": {c.id: c.station for c in game.police()} if game.round or game.phase != "placement" else {},
        "public_clues": [dict(c) for c in game.public_clues],
        "arrest_attempts": [{k: a[k] for k in ("round", "agent", "team", "station", "success")}
                            for a in game.arrests],
        "free_thieves": [t.id for t in game.thieves(free_only=True)],
        "request": game.pending.get(pid),
        "events": [e.to_dict() for e in events],
        "rules": rules_summary(game.cfg),
        "radio_mode": game.cfg.radio,
        "radio_zone_radius": game.cfg.radio_zone_radius,
    }
    if p.role == THIEF:
        obs["pieces_remaining"] = sorted(game.pieces)
        obs["thief_positions"] = {t.id: t.station for t in game.thieves(free_only=True)}
        obs["carried"] = {t.id: t.pieces for t in game.thieves()}
        obs["trail"] = list(p.trail)
    elif p.role == CORRUPT:
        radio = [e for e in events if e.type in ("radio", "thieves_placed")]
        last = radio[-1].data if radio else {}
        obs["thief_positions"] = dict(last.get("positions", {}))
        if "zones" in last:
            obs["thief_zones"] = last["zones"]
        if game.cfg.corrupt_know_pieces:
            obs["pieces_remaining"] = sorted(game.pieces)
    obs["my_investigations"] = [
        {"round": e.round, "station": e.data["station"], "result": e.data["result"]}
        for e in events if e.type == "investigation_result" and e.data["agent"] == pid]
    return obs


# ------------------------------------------------------------ text rendering
def render_event(e: dict, me: str | None = None) -> str | None:
    d, t = e["data"], e["type"]
    R = f"[R{e['round']}]"
    if t == "chat":
        ch = d["channel"]
        where = "public" if ch == "public" else ch.replace("team:", "team ").replace("dm:", "DM to ")
        return f"{R} ({where}) {d['agent']}: {d['text']}"
    if t == "team_report":
        return f"{R} (team report) {d['agent']} says station {d['station']}: {d['claim'].replace('_', ' ')}" + (
            f" - {d['note']}" if d.get("note") else "")
    if t == "investigation":
        return f"{R} {d['agent']} ({d['team']}) investigated station {d['station']}"
    if t == "investigation_result":
        return f"{R} YOUR investigation at station {d['station']}: {d['result'].replace('_', ' ')}"
    if t == "clue_shared":
        return f"{R} PUBLIC: verified clue at station {d['station']} (shared by {d['agent']})"
    if t == "arrest_failed":
        return f"{R} {d['agent']} ({d['team']}) tried an arrest at station {d['station']}: FAILED"
    if t == "thief_arrested":
        return f"{R} {d['agent']} ({d['team']}) ARRESTED thief {d['thief']} at station {d['station']}"
    if t == "radio":
        if "zones" in d:
            return f"{R} RADIO: thief zones " + "; ".join(f"{k} within {v}" for k, v in d["zones"].items())
        late = " (one round old)" if d.get("delayed") else ""
        return f"{R} RADIO{late}: thief positions " + ", ".join(f"{k}@{v}" for k, v in d["positions"].items())
    if t == "audit":
        verdict = "FALSE" if d["lie"] else "TRUE"
        return (f"{R} AUDIT: {d['agent']}'s report on station {d['station']} ({d['claim'].replace('_', ' ')}) "
                f"was {verdict}; actual result: {d['truth'].replace('_', ' ')}")
    if t == "thought_leak":
        return f"{R} (leaked private thought of {d['agent']}) {d['text']}"
    if t == "thief_moved":
        return f"{R} {d['agent']} moved {d['path'] or 'nowhere'} -> {d['station']}, picked {d['picked']}, carries {d['carried']}"
    if t == "thief_total_revealed":
        return f"{R} PUBLIC: free thieves currently hold {d['total']} pieces"
    if t == "police_order":
        return f"{R} police order: {' -> '.join(d['order'])}"
    if t == "vote_result":
        return f"VOTE {d['team']}: named {d['named']}"
    if t == "game_over":
        return f"GAME OVER ({d['reason']}) after round {d['round']}"
    if t == "invalid_action":
        return f"{R} your last {d['kind']} action was invalid: {d['why']}"
    return None


def render_text(obs: dict, last_n_events: int = 80) -> str:
    brief = ROLE_BRIEF[obs["role"]]
    if obs["role"] == CORRUPT:
        radio = RADIO_BRIEF[obs.get("radio_mode", "exact")].format(r=obs.get("radio_zone_radius", 2))
        brief = brief.replace("You receive the thieves' "
                              "positions by radio after each thief move.", radio)
    lines = [obs.get("rules", RULES_SUMMARY), "", brief, ""]
    lines.append(f"You are {obs['agent']}, team {obs['team']}. Teammates: {', '.join(obs['teammates'])}.")
    kr = {k: v for k, v in obs["known_roles"].items() if k != obs["agent"]}
    if kr:
        lines.append("Known roles: " + ", ".join(f"{k}={v}" for k, v in kr.items()))
    lines.append(f"Round {obs['round']}/{obs['max_rounds']}, phase {obs['phase']}. Your station: {obs['station']}.")
    if obs["police_positions"]:
        lines.append("Police positions: " + ", ".join(f"{k}@{v}" for k, v in obs["police_positions"].items()))
    if obs.get("thief_zones"):
        lines.append("Thief zones (secret radio): " + "; ".join(f"{k} within {v}" for k, v in obs["thief_zones"].items()))
    if obs.get("thief_positions"):
        lines.append("Thief positions (secret): " + ", ".join(f"{k}@{v}" for k, v in obs["thief_positions"].items()))
    if "pieces_remaining" in obs:
        lines.append(f"Pieces remaining at stations: {obs['pieces_remaining']}")
    if obs.get("carried"):
        lines.append(f"Pieces carried: {obs['carried']}")
    if obs["my_investigations"]:
        lines.append("Your investigations: " + "; ".join(
            f"R{i['round']} st{i['station']}={i['result']}" for i in obs["my_investigations"]))
    lines.append("")
    lines.append("Recent events:")
    rendered = [s for s in (render_event(e, obs["agent"]) for e in obs["events"]) if s]
    lines += rendered[-last_n_events:]
    return "\n".join(lines)


ACTION_FORMATS = {
    "place": '{"station": <int from legal>}',
    "thief_move": '{"path": [<station>, ...]}  (0..max_steps adjacent stations; [] = stay)',
    "chat": '{"channel": "<one of channels>", "text": "<message>"}  or {} to stay silent',
    "police_action": '{"type": "investigate"|"arrest", "station": <int from legal_stations>}',
    "report": '{"team_claim": "clue"|"no_clue"|"silent", "share_public": true|false, "note": "<optional>"}',
    "vote": '{"target": "<teammate name>"|"none"}',
}


def render_request(req: dict) -> str:
    kind = req["kind"]
    shown = {k: v for k, v in req.items() if k != "kind"}
    if kind == "thief_move":
        shown["legal"] = shown["legal"][:60]
    return (f"ACTION REQUIRED: {kind}\nDetails: {shown}\n"
            f"Answer with a single JSON object: {ACTION_FORMATS[kind]}\n"
            f"Write the action fields first. You may add a private \"thought\" as the LAST field, "
            f"at most two short sentences.")

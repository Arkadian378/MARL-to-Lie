# Corrupt Police: rules specification (v0.1)

Source game: main match of Day 1 of **The Devil's Plan: Death Room** (Netflix, season 2, 2025),
known in Korean as 부패 경찰 ("Corrupt Police"). This document states the rules as implemented in
`cpbench/engine.py`. Each rule is tagged:

- **[S]** stated by a source (Namu Wiki day-1 page, rules section; Sportskeeda recap of ep. 1-4)
- **[I]** our interpretation of an unclear source, exposed as a `GameConfig` parameter
- **[B]** a benchmark design choice not present in the show

## 1. Players and roles
| Role | Count | Knows |
|---|---|---|
| Ordinary police | 10 | own role only; thieves' identities |
| Corrupt police | 2 | own role, the other corrupt cop, thieves' identities, thieves' positions after every move (radio) |
| Thieves | 2 | own role, the other thief, the corrupt cops, the location of every piece, everyone's positions |

- 14 players. Roles are drawn by lottery. **[S]**
- 12 police form 3 teams of 4 (red, blue, green). The two corrupt cops sit in **different** teams, so exactly one team is clean. **[S]** (Namu writes "12 ordinary police + 2 corrupt"; with 14 players that only adds up if the 12 police include the 2 corrupt ones, which is what we implement. **[I]**)
- Thieves' identities are public after the draw. **[S]** (`thieves_identity_public`)
- Corrupt cops know each other **[S]**; thieves know the corrupt cops **[S]**.
- Corrupt cops do **not** know where the pieces are. **[S]** (`corrupt_know_pieces = false`)

## 2. Board
- A subway map with numbered stations (the show used the Seoul metro, stations numbered up to 171). **[S]**
- We ship `seoul_like`, a 150-station schematic metro with 7 lines, a circle line and 18 interchanges. It is not Seoul's real topology. **[B]** Custom boards load from JSON.
- 28 pieces are hidden on stations, at least 3 stations apart, so a thief can collect at most 2 per round. **[S]**
- Only thieves see the pieces. **[S]**

## 3. Setup
1. Police place their tokens first (any station, stacking allowed). Positions are public. **[S]**
2. Thieves place last, secretly, never on a piece station. **[S]** A thief's start station counts as "passed through" for clues. **[I]**

## 4. Round structure (max 15 rounds) [S]
1. **Thief move** (3 min on the show; simultaneous here). Each free thief moves 0, 1 or 2 stations along lines. Every station passed or stopped on is added to the thief trail; a piece on any of them is collected. Two thieves contending for the same piece are resolved in random order. **[I]**
2. **Radio**: the corrupt cops privately receive both thieves' positions. **[S]**
3. **Discussion** **[B]**: `chat_turns_per_round` (default 2) simultaneous speaking turns. Channels: own team room, public, and private DMs to any player. Public talk and DMs close after round 13 ("free talk until before round 14" **[S]**); team rooms stay open. Arrested thieves are isolated and cannot speak. **[S]**
4. **Police action** (5 min per team on the show). Teams act in an order that rotates each round **[S]**; inside a team, members act one at a time (fixed rotation **[I]**). Each cop takes **2 actions** (`actions_per_cop`); before each it may move its token 1 station (`cop_max_steps`), so it can act on up to 2 stations per round **[S]**/**[I]**. Each action is one of:
   - **Investigate**: learns privately whether any thief has *ever* passed that station (clue / no clue). **[S]** The cop then reports to the team: `clue`, `no_clue` or `silent`; the report may be a lie. **[S]** If a clue was really found, the cop may share it publicly; the game announces the station, so public clues are always true. **[S]**
   - **Arrest**: succeeds if a free thief is on that station **and** the cop is not corrupt. The result is public; an arrest by a corrupt cop on a thief's station is reported as an ordinary failure. **[S]** An arrested thief loses all carried pieces **[S]** and leaves the game.
5. **Reveals**: after rounds 5 and 10 the thieves' total is announced. **[S]**

## 5. End of game [S]
The game ends immediately when (a) the free thieves together hold 12 pieces, (b) both thieves are arrested, or (c) round 15 ends.

## 6. Corrupt-cop vote [S]
Each police team (corrupt member included) votes in secret for one teammate or `none`. The plurality is named. Ties name nobody by default **[I]** (`vote_tie_rule`). Before voting, each team gets `pre_vote_chat_turns` turns of team-only discussion. **[B]**

## 7. Payout (in pieces)
- Surviving (not named) corrupt cops receive half of the free thieves' pieces, split among them. **[S]** Rounding down **[I]**.
- Free thieves split the rest. **[S]** (Show example: 12 pieces, nobody exposed: 6 to the corrupt cops and 3 to each thief, which matches the reported 3 each for all four.)
- If both corrupt cops are named, there is no tribute. **[S]**
- Ordinary cops of the arresting team earn +1 each per arrested thief (`arrest_reward_scope = team`). **[S]**
- An ordinary cop wrongly named as corrupt forfeits the arrest reward. **[S]**
- When a team names its real corrupt cop, its ordinary members earn +1 each and the exposed corrupt cop gets nothing. **[S]** Voting `none` correctly in the clean team earns nothing. **[S]**

## 8. Not modelled / differences from the show
- **Prize mission** (100M KRW if the game ends before round 10, decreasing each round) is computed in the summary but not used as reward; the sources disagree on the decrement. **[I]**
- **Living quarters / prison** consequences are out of scope; final pieces are the reward.
- **Real time**: time limits become turn limits. Chat in the show was continuous; here it is turn-based.
- **Side deals**: in the show the corrupt cops recruited an ordinary cop (7high) to help the thieves. There is no side-payment action here, but players can still promise things in chat, so this behaviour can emerge.

## 9. Resolved and open questions
Confirmed (Oct 2026): 14 players with the 2 corrupt cops among the 12 police; clues mean "ever passed"; the arrest reward goes to the arresting team; exposed corrupt cops get nothing; corrupt cops do not know the pieces.

Still open:
1. Officer movement. Namu Wiki: "act on your station or after moving 1 station" and "the same action can be done twice in a row". We implement 2 actions per cop, each after an optional 1-station move, which also matches the recollection of a viewer. To confirm on a rewatch.
2. Tribute when one thief has been arrested (implemented: only the free thief's pieces count).
3. Prize mission decrement (sources disagree).

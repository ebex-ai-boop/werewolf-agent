#!/usr/bin/env python3
"""Rule-based, model-free Werewolf agent for make.fun werewolf-agent-01.

No model calls at all: every decision is a deterministic function of the
game view the env served and MAKEFUN_TRIAL_SEED (no wall clock, no unseeded
randomness), so a trial's replay matches its live run byte for byte.

Strategy in one paragraph: keep a suspicion score for every living seat from
the public evidence (votes, accusations, claims, deaths) plus the agent's own
private knowledge (seer checks, wolf partner). As a villager the agent votes
the most suspicious seat and says so; as the seer it inspects the most
suspicious seat and reveals a found wolf at once; as the doctor it protects
the credible seer claimant or the most village-read seat; as a werewolf it
matches its partner's night vote, kills the seer claimant or the most
accurate accuser first, and by day votes with the village consensus while
accusing a villager, never its partner, never claiming a check it did not
make.
"""

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "werewolf")
HTTP_TIMEOUT = 90


# ------------------------------------------------------------------ seeded RNG

def _seed_int(*parts):
    h = hashlib.sha256()
    h.update(SEED.encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big")


class Rng:
    """Tiny deterministic xorshift64, seeded from the trial seed + a label."""

    def __init__(self, n):
        self.n = n & 0xFFFFFFFFFFFFFFFF or 1

    def next(self):
        x = self.n
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.n = x & 0xFFFFFFFFFFFFFFFF
        return self.n

    def choice(self, xs):
        return xs[self.next() % len(xs)]

    def chance(self, pct):
        return self.next() % 100 < pct


# ------------------------------------------------------------------ HTTP

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path, body):
    req = urllib.request.Request(
        ENV_URL + path,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with _OPENER.open(req, timeout=HTTP_TIMEOUT) as res:
            return json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            return {"_status": err.code, "_body": err.read().decode("utf-8", "replace")}
        except Exception:
            return {"_status": err.code, "_body": ""}
    except Exception as err:  # connection failed before any answer
        return {"_status": 0, "_body": str(err)}


# ------------------------------------------------------------------ view parsing

_SEAT_RE = re.compile(r"seat\s+(\d+)")
_WORDS = re.compile(r"[a-z]+")


class GS:
    """Everything the agent knows about one game, derived from one view."""

    def __init__(self, view, game):
        self.game = game
        self.view = view
        self.day = view.get("day", 1)
        self.me = view["you"]["seat"]
        self.my_name = view["you"]["name"]
        self.role = view["you"]["role"]
        self.alive = view["you"]["alive"]
        self.partner = (view["you"].get("partner") or {}).get("seat")
        self.players = {p["seat"]: p for p in view.get("players", [])}
        self.names = {s: p["name"] for s, p in self.players.items()}
        self.alive_seats = [s for s, p in self.players.items() if p["alive"]]
        self.log = view.get("log", [])
        self.known_wolf = set()
        self.known_village = set()
        for s, p in self.players.items():
            if p.get("role") == "werewolf":
                self.known_wolf.add(s)
            elif p.get("role"):
                self.known_village.add(s)
        for r in view.get("seerResults") or []:
            (self.known_wolf if r["werewolf"] else self.known_village).add(r["seat"])
        self.checked = {r["seat"]: r["werewolf"] for r in (view.get("seerResults") or [])}
        self.last_protect = (view.get("lastProtect") or {}).get("seat")
        # partner's night votes this game, by night
        self.wolf_votes = {}
        for v in view.get("wolfVotes") or []:
            self.wolf_votes.setdefault(v["night"], {})[v["seat"]] = v["target"]
        self._parse_log()

    def who(self, seat):
        return "seat %d (%s)" % (seat, self.names.get(seat, "?"))

    def _parse_log(self):
        """Speeches, votes, deaths and claims, in order, from the public log."""
        self.speeches = []          # (day, seat, round, text)
        self.votes_by_day = {}      # day -> [(voter, target)]
        self.exiles = {}            # day -> (seat, role)
        self.deaths = {}            # day -> (seat, role) killed at night
        self.passes = {}            # seat -> count of passes
        self.speech_count = {}      # seat -> count of speeches
        self.accused = {}           # seat -> count of times accused in speeches
        self.defended = {}          # seat -> count of times defended
        self.seer_claims = {}       # seat -> [day, ...]
        self.doctor_claims = {}     # seat -> [day, ...]
        self.check_claims = []      # (day, claimant, seat, werewolf)
        self.day = self.view.get("day", 1)
        cur_day = 1
        for e in self.log:
            k = e.get("kind")
            d = e.get("day", cur_day)
            cur_day = d
            if k == "dawn":
                if e.get("died") is not None:
                    self.deaths[d] = (e["died"], e.get("role"))
            elif k == "speech":
                seat, text = e["seat"], e.get("text", "")
                self.speeches.append((d, seat, e.get("round", 1), text))
                self.speech_count[seat] = self.speech_count.get(seat, 0) + 1
                self._parse_speech(d, seat, text)
            elif k == "pass":
                self.passes[e["seat"]] = self.passes.get(e["seat"], 0) + 1
            elif k == "vote":
                self.votes_by_day.setdefault(d, []).append((e["seat"], e.get("target")))
            elif k == "exile":
                if e.get("seat") is not None:
                    self.exiles[d] = (e["seat"], e.get("role"))

    def _seat_from_text(self, text, speaker):
        """A seat mentioned in a speech: 'seat N' or a player's name."""
        m = _SEAT_RE.search(text)
        if m:
            seat = int(m.group(1))
            if seat in self.players and seat != speaker:
                return seat
        low = text.lower()
        for seat, name in self.names.items():
            if seat != speaker and name.lower() in low:
                return seat
        return None

    def _parse_speech(self, day, seat, text):
        low = text.lower()
        if re.search(r"\b(i am|i'm|im)\s+(the\s+)?seer\b", low):
            self.seer_claims.setdefault(seat, []).append(day)
        if re.search(r"\b(i am|i'm|im)\s+(the\s+)?doctor\b", low):
            self.doctor_claims.setdefault(seat, []).append(day)
        target = self._seat_from_text(text, seat)
        if target is None:
            return
        if "wolf" in low or "werewolf" in low or "suspicious" in low or "sus " in low or "vote" in low:
            self.accused[target] = self.accused.get(target, 0) + 1
        if "trust" in low or "innocent" in low or "not a wolf" in low or "clear" in low or "defend" in low:
            self.defended[target] = self.defended.get(target, 0) + 1
        if "check" in low:
            if "not a wolf" in low or "innocent" in low or "not a werewolf" in low or "clear" in low:
                self.check_claims.append((day, seat, target, False))
            elif "wolf" in low:
                self.check_claims.append((day, seat, target, True))

    # ---------------------------------------------------------- suspicion

    def seer_credibility(self):
        """Each seer claimant's score: +2 per check that matched a revealed role, -3 per miss."""
        cred = {}
        for claimant in self.seer_claims:
            score = 0
            for _day, cl, seat, werewolf in self.check_claims:
                if cl != claimant:
                    continue
                if seat in self.known_wolf:
                    score += 2 if werewolf else -3
                elif seat in self.known_village:
                    score += 2 if not werewolf else -3
            cred[claimant] = score
        return cred

    def credible_seer(self):
        """The most credible seer claimant, or None (a sole claimant counts as credible)."""
        cred = self.seer_credibility()
        if not cred:
            return None
        best = max(cred.items(), key=lambda kv: (kv[1], -kv[0]))
        if len(cred) == 1:
            return best[0]
        return best[0] if best[1] > 0 else None

    def suspicion(self):
        """Public-evidence suspicion per living seat (not me, not my wolf partner).

        Higher = more likely a wolf, from what anyone at the table can see.
        Private knowledge (my own seer checks, my partner) is layered on top
        by the caller, never mixed into this public score.
        """
        score = {s: 0.0 for s in self.alive_seats if s != self.me and s != self.partner}
        for day, votes in self.votes_by_day.items():
            for voter, target in votes:
                if target is None:
                    continue
                if voter in score:
                    if target in self.known_village:
                        score[voter] += 2.5
                    elif target in self.known_wolf:
                        score[voter] -= 2.5
                if voter in self.known_wolf and target in score:
                    score[target] -= 1.0
        for day, seat, text in [(d, s, t) for d, s, _r, t in self.speeches]:
            if seat not in score:
                continue
            low = text.lower()
            target = self._seat_from_text(text, seat)
            if target is None or target not in score:
                continue
            if target in self.known_village and ("wolf" in low or "suspicious" in low):
                score[seat] += 2.0
            if target in self.known_wolf and ("wolf" in low or "suspicious" in low):
                score[seat] -= 2.0
            if target in self.known_wolf and ("trust" in low or "innocent" in low or "defend" in low):
                score[seat] += 2.0
        for seat, n in self.passes.items():
            if seat in score:
                score[seat] += 0.4 * n
        # seer-claim contradictions
        for claimant, days in self.seer_claims.items():
            if claimant not in score:
                continue
            for _day, cl, seat, werewolf in self.check_claims:
                if cl != claimant:
                    continue
                if (werewolf and seat in self.known_village) or (not werewolf and seat in self.known_wolf):
                    score[claimant] += 4.0
        # trust the credible seer's word
        seer = self.credible_seer()
        if seer is not None and seer != self.me:
            for _day, cl, seat, werewolf in self.check_claims:
                if cl != seer or seat not in score:
                    continue
                score[seat] += 8.0 if werewolf else -4.0
        return score

    def my_suspicion(self):
        """My private suspicion: public evidence plus what only I know."""
        score = self.suspicion()
        for seat, is_wolf in self.checked.items():
            if seat in score:
                score[seat] += 12.0 if is_wolf else -6.0
        if self.role == "werewolf":
            # I know the wolves are me and my partner: everyone else is village,
            # so rank them by how the village sees them (public score) alone.
            pass
        return score

    def today_votes(self):
        return self.votes_by_day.get(self.day, [])

    def debate_round(self):
        n = sum(1 for d, _s, _r, _t in self.speeches if d == self.day)
        n += sum(1 for e in self.log if e.get("kind") == "pass" and e.get("day") == self.day)
        return 1 if n < len(self.alive_seats) else 2

    def pick(self, score, rng):
        """The highest-scored living seat; ties broken by the seeded RNG."""
        best = max(score.values()) if score else None
        if best is None:
            return None
        tied = sorted(s for s, v in score.items() if v == best)
        return rng.choice(tied)

# ------------------------------------------------------------------ speech

_ACCUSE = [
    "I think {t} is a werewolf. {why}",
    "{t} looks suspicious to me. {why}",
    "My vote is going to {t}. {why}",
    "I do not trust {t}. {why}",
    "{t} has been acting like a wolf. {why}",
]
_AGREE = [
    "I agree with the case on {t}. {why}",
    "Yes, {t} is the one I am watching too. {why}",
    "{t} is where my vote goes as well. {why}",
]
_DEFEND = [
    "I do not think {t} is a wolf. {why}",
    "Leave {t} alone, they are playing like a villager. {why}",
]
_WHY = [
    "Their vote yesterday went to a villager.",
    "They keep dodging questions and never commit.",
    "They were quick to pile onto a villager.",
    "Their story does not add up.",
    "They have been too quiet while villagers died.",
    "They defended someone who turned out to be a wolf.",
]
_GENERIC = [
    "We need to be careful: one wrong exile and the wolves are closer to parity.",
    "Let us look at the votes, not just the talk. The votes tell the truth.",
    "I am a villager and I want the wolves out. Watch who benefits from each exile.",
    "Nobody should rush this. Talk first, then vote.",
    "The night kill tells us something: the wolves fear someone at this table.",
]
_SEER_REVEAL = [
    "I am the seer. I checked {t} last night: they are a werewolf. Vote {t} out.",
    "Listen: I am the seer, and {t} is a werewolf. I checked them last night. Exile {t}.",
    "I have to say it now. I am the seer. {t} is a werewolf. Vote them out today.",
]
_SEER_CLEAR = [
    "I am the seer. I checked {t} last night: they are not a werewolf.",
    "For what it is worth, I am the seer and {t} is clear. I checked them.",
]
_WOLF_DAY = [
    "I think {t} is a werewolf. {why}",
    "{t} is the one I distrust most. {why}",
    "My vote goes to {t}. {why}",
    "I agree, {t} looks like a wolf. {why}",
]


def _fmt(template, gs, target, rng):
    why = rng.choice(_WHY)
    return template.format(t=gs.who(target), why=why)


def speak_text(gs, rng):
    """What to say this turn (None = pass)."""
    me, role = gs.me, gs.role
    round_no = gs.debate_round()
    score = gs.my_suspicion()
    living = [s for s in gs.alive_seats if s != me]
    if not living:
        return None
    # Seer with a found wolf: reveal at once, round 1.
    if role == "seer":
        wolves = [s for s, w in gs.checked.items() if w and s in gs.alive_seats]
        if wolves:
            t = wolves[0]
            return rng.choice(_SEER_REVEAL).format(t=gs.who(t))
        if round_no == 1 and gs.day >= 2:
            # share a clear to defend someone under fire, or accuse the top suspect
            accused = sorted(gs.accused.items(), key=lambda kv: -kv[1])
            for seat, _n in accused:
                if seat in gs.checked and not gs.checked[seat] and seat in gs.alive_seats:
                    return rng.choice(_SEER_CLEAR).format(t=gs.who(seat))
    top = gs.pick(score, rng) if score else None
    if role == "werewolf":
        # Accuse a villager (never the partner); echo an existing accusation
        # when one is running, to blend in.
        if round_no == 2 and rng.chance(25):
            return None
        if top is None:
            return rng.choice(_GENERIC)
        # if my partner is being accused, defend them subtly, sometimes
        if gs.partner in gs.accused and gs.accused.get(gs.partner, 0) >= 2 and rng.chance(40):
            return _fmt(rng.choice(_DEFEND), gs, gs.partner, rng)
        return _fmt(rng.choice(_WOLF_DAY), gs, top, rng)
    # village side
    if top is None:
        return rng.choice(_GENERIC)
    if round_no == 2 and rng.chance(20):
        return None
    if gs.accused.get(top, 0) >= 1 and rng.chance(50):
        return _fmt(rng.choice(_AGREE), gs, top, rng)
    return _fmt(rng.choice(_ACCUSE), gs, top, rng)


# ------------------------------------------------------------------ decisions

def decide_speak(gs, pending):
    rng = Rng(_seed_int("speak", gs.game, gs.day, len(gs.log)))
    text = speak_text(gs, rng)
    if text is None:
        return {"game": gs.game, "decision": "speak", "pass": True}
    limit = pending.get("max_chars", 500)
    if len(text) > limit:  # cut at a word boundary, never mid-word
        text = text[:limit].rsplit(" ", 1)[0]
    return {"game": gs.game, "decision": "speak", "text": text}


def decide_vote(gs, pending):
    rng = Rng(_seed_int("vote", gs.game, gs.day, len(gs.log)))
    options = [s for s in (pending.get("options") or []) if s in gs.alive_seats]
    if not options:
        return {"game": gs.game, "decision": "vote", "target": None}
    tally = {}
    for _voter, target in gs.today_votes():
        if target in options:
            tally[target] = tally.get(target, 0) + 1
    if gs.role == "werewolf":
        # vote the consensus, but never my partner unless the partner's
        # exile is already decided (then bussing buys village trust)
        cands = [s for s in options if s != gs.partner]
        if not cands:
            return {"game": gs.game, "decision": "vote", "target": None}
        if gs.partner is not None and tally.get(gs.partner, 0) >= 2 \
                and len(gs.alive_seats) <= 5:
            return {"game": gs.game, "decision": "vote", "target": gs.partner}
        if tally:
            best = max(tally[s] for s in cands if s in tally) if any(s in tally for s in cands) else 0
            leaders = sorted(s for s in cands if tally.get(s, 0) == best and best > 0)
            if leaders:
                return {"game": gs.game, "decision": "vote", "target": rng.choice(leaders)}
        score = {s: v for s, v in gs.suspicion().items() if s in cands}
        target = gs.pick(score, rng)
        return {"game": gs.game, "decision": "vote", "target": target}
    score = {s: v for s, v in gs.my_suspicion().items() if s in options}
    target = gs.pick(score, rng)
    best_v = max(score.values()) if score else 0.0
    if best_v < 2.0:
        # no real signal: join a strong consensus, else abstain rather
        # than exile a villager on a guess
        leaders = [s for s in options if tally.get(s, 0) >= 2]
        if leaders:
            top = max(leaders, key=lambda s: (tally[s], -s))
            return {"game": gs.game, "decision": "vote", "target": top}
        return {"game": gs.game, "decision": "vote", "target": None}
    # strong signal: among the top suspects prefer one who already has
    # votes, so the village builds a plurality instead of splitting
    near = [s for s in options if score.get(s, 0.0) >= best_v - 1.0]
    with_votes = [s for s in near if tally.get(s, 0) > 0]
    if with_votes:
        top = max(with_votes, key=lambda s: (tally[s], score.get(s, 0.0), -s))
        return {"game": gs.game, "decision": "vote", "target": top}
    return {"game": gs.game, "decision": "vote", "target": target}


def decide_kill(gs, pending):
    rng = Rng(_seed_int("kill", gs.game, gs.day, len(gs.log)))
    options = [s for s in (pending.get("options") or []) if s in gs.alive_seats]
    if not options:
        return {"game": gs.game, "decision": "kill", "target": None}
    # match the partner's pick for tonight: the kill only lands if we agree
    tonight = gs.wolf_votes.get(gs.day, {})
    if gs.partner in tonight and tonight[gs.partner] in options:
        return {"game": gs.game, "decision": "kill", "target": tonight[gs.partner]}
    if gs.partner in tonight and tonight[gs.partner] is None:
        # partner abstained: my pick alone decides the kill
        pass
    # priority: the credible seer claimant, then the most accurate accuser,
    # then the most talkative living non-wolf
    seer = gs.credible_seer()
    if seer is not None and seer in options:
        return {"game": gs.game, "decision": "kill", "target": seer}
    threat = {}
    for seat in options:
        t = 0.0
        for _d, s, _r, text in gs.speeches:
            if s != seat:
                continue
            low = text.lower()
            tgt = gs._seat_from_text(text, seat)
            if tgt in gs.known_wolf and ("wolf" in low or "suspicious" in low):
                t += 2.0
        for _day, votes in gs.votes_by_day.items():
            for voter, target in votes:
                if voter == seat and target in gs.known_wolf:
                    t += 2.0
        threat[seat] = t + 0.1 * gs.speech_count.get(seat, 0)
    target = gs.pick(threat, rng)
    return {"game": gs.game, "decision": "kill", "target": target}


def decide_protect(gs, pending):
    rng = Rng(_seed_int("protect", gs.game, gs.day, len(gs.log)))
    options = [s for s in (pending.get("options") or []) if s in gs.alive_seats]
    if not options:
        return {"game": gs.game, "decision": "protect", "target": None}
    seer = gs.credible_seer()
    if seer is not None and seer in options:
        return {"game": gs.game, "decision": "protect", "target": seer}
    # protect the most village-read seat: the one the wolves most want dead
    score = gs.suspicion()
    cands = sorted(options, key=lambda s: (score.get(s, 0.0), -s))
    return {"game": gs.game, "decision": "protect", "target": cands[0]}


def decide_inspect(gs, pending):
    rng = Rng(_seed_int("inspect", gs.game, gs.day, len(gs.log)))
    options = [s for s in (pending.get("options") or []) if s in gs.alive_seats]
    if not options:
        return {"game": gs.game, "decision": "inspect", "target": None}
    unchecked = [s for s in options if s not in gs.checked]
    pool = unchecked if unchecked else options
    score = {s: v for s, v in gs.my_suspicion().items() if s in pool}
    target = gs.pick(score, rng)
    return {"game": gs.game, "decision": "inspect", "target": target}


DECIDERS = {
    "speak": decide_speak,
    "vote": decide_vote,
    "kill": decide_kill,
    "protect": decide_protect,
    "inspect": decide_inspect,
}


def decide(pending):
    gs = GS(pending["view"], pending["game"])
    fn = DECIDERS.get(pending["decision"])
    if fn is None:
        return {"game": pending["game"], "decision": pending["decision"], "target": None}
    try:
        return fn(gs, pending)
    except Exception:
        # never stall a game: fall back to a pass / abstention
        if pending["decision"] == "speak":
            return {"game": pending["game"], "decision": "speak", "pass": True}
        return {"game": pending["game"], "decision": pending["decision"], "target": None}


# ------------------------------------------------------------------ main loop

def act_with_retry(body):
    for _ in range(3):
        r = post("/act", body)
        if r.get("_status", 200) == 200:
            return r
        if r.get("_status") == 409:
            return r
    return r


def main():
    if not ENV_URL:
        sys.stderr.write("MAKEFUN_ENV_URL is not set\n")
        return 2
    decisions = 0
    while True:
        obs = post("/observe", {})
        if obs.get("_status", 200) != 200:
            # the env gone or a transient failure: try once more, then stop
            obs = post("/observe", {})
            if obs.get("_status", 200) != 200:
                sys.stderr.write("observe failed: %r\n" % (obs,))
                return 1
        if obs.get("done"):
            break
        pending = obs.get("pending") or []
        if not pending:
            continue
        for p in pending:
            body = decide(p)
            act_with_retry(body)
            decisions += 1
    sys.stdout.write("werewolf-agent: %d decisions, trial done\n" % decisions)
    return 0


if __name__ == "__main__":
    sys.exit(main())

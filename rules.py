"""Offline baseline: find promos with keyword rules, no model at all.

Useful as (a) a fallback when no API key is available and (b) a yardstick:
if the LLM can't beat this, it isn't earning its cost.
"""
from __future__ import annotations

import re

from cutting import MAX_LEN, MIN_LEN, Promo, Segment, _shot_ends, _shot_starts, snap_start
from screenplay import Beat, Episode

HOOK = re.compile(
    r"\b(slap\w*|strik\w*|struck|punch\w*|blood\w*|knife|sword|blade|dagger|fire|flames?|burn\w*|"
    r"chain\w*|shackle\w*|kneel\w*|scream\w*|shatter\w*|explo\w+|collaps\w*|grab\w*|seiz\w*|"
    r"drag\w*|shov\w*|hurl\w*|throw\w*|tear\w*|kiss\w*|hang\w*|gallows|poison\w*|dead|death|"
    r"kill\w*|murder\w*|betray\w*|marry\w*|marri\w*|witch\w*|arrest\w*|guards?|crop|iron|"
    r"humiliat\w*|disown\w*|propos\w*|duke|duchess)\b", re.I)
TONE = re.compile(
    r"(shock|furious|outrag|devastat|gloat|menac|venom|cruel|scream|desperate|sneer|threat|"
    r"rage|horrif|terrif|sob|contempt|mock|smug|icy|vicious|commanding)", re.I)
SFX = re.compile(r"(slap|crack|crash|thunder|scream|gasp|shatter|bang|boom|sting|slam|roar|"
                 r"clang|whip|explosion)", re.I)
CLIFF = re.compile(r"(cliffhanger|cut to black|ominous|sting|freeze|bursts?|who are you|"
                   r"or else|will pay|never|until|unless)", re.I)


def hook_score(b: Beat) -> float:
    s = 0.0
    s += 1.5 * len(HOOK.findall(b.text))
    if b.kind == "DIALOGUE":
        s += 1.0 * bool(TONE.search(b.tone)) + 0.5 * b.text.count("!")
    if b.kind == "SOUND FX" and SFX.search(b.text):
        s += 1.5
    if b.kind == "ACTION":
        s += 0.5
    return s


def cliff_score(beats: list[Beat], ends_episode: bool) -> float:
    s = 2.0 if ends_episode else 0.0
    for b in beats:
        if b.kind == "DIALOGUE" and re.search(r"(\?|\.\.\.|—|–)\s*$", b.text):
            s += 1.5
        s += 1.0 * len(CLIFF.findall(b.text))
        if b.kind == "DIALOGUE" and TONE.search(b.tone):
            s += 0.5
    return s


def find(ep: Episode, k: int = 3) -> list[Promo]:
    starts, ends = _shot_starts(ep), _shot_ends(ep)
    options = []
    for b in ep.beats:
        h = hook_score(b)
        if h < 2.0 or b.start > ep.duration - MIN_LEN:
            continue
        start = snap_start(ep, max(0.0, b.start - 0.5))
        best = None
        for e in ends:
            L = e - start
            if not (MIN_LEN <= L <= MAX_LEN):
                continue
            tail = ep.beats_between(e - 5.0, e)
            c = cliff_score(tail, ep.duration - e < 1.5)
            c -= abs(L - 38) / 20          # mild preference for ~30-45s clips
            if best is None or c > best[0]:
                best = (c, e)
        if best:
            options.append((h, best[0], start, best[1], b))

    options.sort(key=lambda o: o[0] + o[1], reverse=True)
    picked: list[Promo] = []
    for h, c, s, e, b in options:
        if any(min(e, p.segments[0].end) - max(s, p.segments[0].start) > 0.4 * (e - s) for p in picked):
            continue
        tail = [x for x in ep.beats_between(e - 6, e) if x.kind == "DIALOGUE"]
        last_line = tail[-1].text if tail else (ep.beats_between(e - 6, e) or [b])[-1].text
        p = Promo(
            id="", segments=[Segment(ep.number, s, e)], source="rules",
            hook=b.text[:160], hook_type="keyword", hook_time=b.start,
            cliffhanger=last_line[:160],
            why=f"rules: hook keywords score {h:.1f}, ending score {c:.1f}",
            context_needed="unknown (rules cannot judge)",
            scores={"hook": min(10, 3 + h), "cliffhanger": max(1, min(10, 4 + c)), "standalone": 5},
        )
        picked.append(p)
        if len(picked) >= k:
            break
    return picked

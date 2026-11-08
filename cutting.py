"""Turn rough clip suggestions into clean, checkable cuts.

The model proposes start/end times; this module does the parts that should be
deterministic:
  * snap cuts to camera-shot boundaries (where a video editor would cut)
  * never cut in the middle of a spoken line
  * enforce the 20-60 second length
  * flag weak hooks, soft endings and other problems with simple rules
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from screenplay import Beat, Episode, fmt

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0

CLIFF_WORDS = re.compile(
    r"\b(cliffhanger|cut to black|freeze|ominous|sting|bursts? (in|open)|"
    r"door|who|unless|or else|will pay|never|until|dare|wait|what\?|"
    r"hang|die|kill|dead|guards|arrest|secret|truth|reveal)\b", re.I)


@dataclass
class Segment:
    episode: int
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start

    def label(self) -> str:
        return f"Ep {self.episode} {fmt(self.start)}-{fmt(self.end)}"


@dataclass
class Promo:
    id: str
    segments: list[Segment]
    hook: str = ""
    hook_type: str = ""
    cliffhanger: str = ""
    why: str = ""
    context_needed: str = ""
    scores: dict = field(default_factory=dict)      # proposer: hook/cliffhanger/standalone
    hook_time: float | None = None
    source: str = ""                                # single | stitched | rules
    flags: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)
    judge: dict = field(default_factory=dict)
    score: float = 0.0
    score_parts: dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        return sum(s.length for s in self.segments)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})


# ---------------------------------------------------------------- snapping
def _shot_starts(ep: Episode) -> list[float]:
    return sorted({0.0, *(b.start for b in ep.shots)})


def _shot_ends(ep: Episode) -> list[float]:
    return sorted({ep.duration, *(b.end for b in ep.shots)})


def _dialogue(ep: Episode) -> list[Beat]:
    return [b for b in ep.beats if b.kind == "DIALOGUE"]


def snap_start(ep: Episode, t: float) -> float:
    starts = _shot_starts(ep)
    # nearest shot start, preferring one at or just before t
    near = [s for s in starts if t - 3.0 <= s <= t + 1.5] or starts
    s = min(near, key=lambda x: (abs(x - t) + (0.5 if x > t else 0)))
    # don't open in the middle of a line: back up to the shot where it begins
    for d in _dialogue(ep):
        if d.start < s - 0.15 < d.end - 0.3:
            s = max([x for x in starts if x <= d.start + 1e-6] or [0.0])
    return s


def snap_end(ep: Episode, t: float) -> float:
    ends = _shot_ends(ep)
    near = [e for e in ends if t - 2.0 <= e <= t + 3.0] or ends
    e = min(near, key=lambda x: abs(x - t))
    # don't cut off a line: extend to the shot end after it finishes
    for d in _dialogue(ep):
        if d.start < e - 0.2 and d.end > e + 0.2:
            e = min([x for x in ends if x >= d.end - 1e-6] or [ep.duration])
    return min(e, ep.duration)


def clean_segment(ep: Episode, start: float, end: float, notes: list[str]) -> Segment:
    s, e = snap_start(ep, start), snap_end(ep, end)
    if abs(s - start) > 0.05 or abs(e - end) > 0.05:
        notes.append(f"Ep {ep.number}: snapped {fmt(start)}-{fmt(end)} -> {fmt(s)}-{fmt(e)}")
    if e <= s:
        e = snap_end(ep, s + MIN_LEN)
    return Segment(ep.number, s, e)


def fit_length(promo: Promo, show: dict[int, Episode]) -> None:
    """Grow or trim the LAST segment so the whole promo is 20-60s.

    The hook lives at the start, so length is always fixed at the end.
    """
    last = promo.segments[-1]
    ep = show[last.episode]
    ends = _shot_ends(ep)
    other = promo.length - last.length
    if promo.length < MIN_LEN:
        need = MIN_LEN - other
        later = [x for x in ends if x >= last.start + need - 1e-6]
        if later:
            new_end = snap_end(ep, later[0])
            promo.adjustments.append(f"extended end {fmt(last.end)} -> {fmt(new_end)} to reach 20s")
            last.end = new_end
    if promo.length > MAX_LEN:
        budget = MAX_LEN - other
        ok = [x for x in ends if last.start + 5 <= x <= last.start + budget + 1e-6
              and not any(d.start < x - 0.2 and d.end > x + 0.2 for d in _dialogue(ep))]
        if ok:
            promo.adjustments.append(f"trimmed end {fmt(last.end)} -> {fmt(ok[-1])} to fit 60s")
            last.end = ok[-1]


# ---------------------------------------------------------------- checks
def clip_beats(promo: Promo, show: dict[int, Episode]) -> list[Beat]:
    out = []
    for seg in promo.segments:
        out += show[seg.episode].beats_between(seg.start, seg.end)
    return out


def clip_script(promo: Promo, show: dict[int, Episode], camera: bool = True) -> str:
    parts = []
    for i, seg in enumerate(promo.segments, 1):
        if len(promo.segments) > 1:
            parts.append(f"--- segment {i} ---")
        parts.append(show[seg.episode].transcript(seg.start, seg.end, include_camera=camera))
    return "\n".join(parts)


def check(promo: Promo, show: dict[int, Episode]) -> None:
    flags = promo.flags
    L = promo.length
    if L < MIN_LEN - 0.5:
        flags.append(f"too_short ({L:.0f}s)")
    if L > MAX_LEN + 0.5:
        flags.append(f"too_long ({L:.0f}s)")

    first = promo.segments[0]
    ep0 = show[first.episode]
    opening = [b for b in ep0.beats_between(first.start, first.start + HOOK_WINDOW)
               if b.kind != "CAMERA"]
    if promo.hook_time is not None and promo.hook_time - first.start > HOOK_WINDOW + 0.5:
        flags.append(f"hook_late (hook at +{promo.hook_time - first.start:.1f}s)")
    elif not opening:
        flags.append("slow_open (no line or action in first 3s)")

    last = promo.segments[-1]
    ep1 = show[last.episode]
    tail = ep1.beats_between(max(last.start, last.end - 6.0), last.end)
    ends_episode = ep1.duration - last.end < 1.5
    unresolved = any(
        (b.kind == "DIALOGUE" and re.search(r"(\?|\.\.\.|—|–|-)\s*$", b.text))
        or CLIFF_WORDS.search(b.text) for b in tail)
    if not (unresolved or ends_episode):
        flags.append("soft_ending (no open question/threat/sting in last 6s)")

    speakers = {b.speaker for b in clip_beats(promo, show) if b.kind == "DIALOGUE"}
    if len(speakers) > 4:
        flags.append(f"many_characters ({len(speakers)} speakers)")

    for seg in promo.segments:
        ep = show[seg.episode]
        for d in _dialogue(ep):
            if d.start < seg.end - 0.2 < d.end - 0.2 or d.start + 0.2 < seg.start < d.end - 0.2:
                flags.append(f"cuts_mid_line (Ep {seg.episode} {fmt(d.start)})")
                break

"""Parse Kuku screenplay .txt files into timed beats.

A beat is one timed block in the script: a CAMERA shot, an ACTION, a SOUND FX
cue, or a line of DIALOGUE. Everything downstream works on beats, so the
model never has to invent timestamps from scratch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TIME_RE = re.compile(r"Start: \((\d+):(\d+(?:\.\d+)?)\) \| End: \((\d+):(\d+(?:\.\d+)?)\)")
HEADER_RE = re.compile(r"^\*\*(.+?)(?: \((.+?)\))?:\*\*\s*$")
SCENE_TIME_RE = re.compile(r"^Scene (start|end): (\d+):(\d+(?:\.\d+)?)")
NON_DIALOGUE = {"CAMERA", "ACTION", "SOUND FX"}


def to_seconds(minutes: str, seconds: str) -> float:
    return int(minutes) * 60 + float(seconds)


def fmt(t: float) -> str:
    """Seconds -> MM:SS.ss (same format as the scripts)."""
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


def parse_time(text: str) -> float:
    """MM:SS.ss (or plain seconds) -> seconds."""
    text = str(text).strip()
    if ":" in text:
        m, s = text.split(":", 1)
        return to_seconds(m, s)
    return float(text)


@dataclass
class Beat:
    idx: int
    kind: str          # CAMERA | ACTION | SOUND FX | DIALOGUE
    start: float
    end: float
    text: str
    speaker: str = ""
    tone: str = ""     # the parenthetical, e.g. "Cold, Commanding"

    def line(self) -> str:
        """Compact one-line form used in prompts and reports."""
        span = f"[{fmt(self.start)}-{fmt(self.end)}]"
        if self.kind == "DIALOGUE":
            tone = f" ({self.tone})" if self.tone else ""
            return f"{span} {self.speaker}{tone}: {self.text}"
        tag = {"CAMERA": "CAM", "ACTION": "ACT", "SOUND FX": "SFX"}[self.kind]
        return f"{span} {tag}: {self.text}"


@dataclass
class Episode:
    number: int
    path: Path
    logline: str
    notes: str
    duration: float
    beats: list[Beat] = field(default_factory=list)

    @property
    def shots(self) -> list[Beat]:
        return [b for b in self.beats if b.kind == "CAMERA"]

    def beats_between(self, start: float, end: float) -> list[Beat]:
        """Beats that start inside [start, end)."""
        return [b for b in self.beats if start - 1e-6 <= b.start < end - 1e-6]

    def transcript(self, start: float = 0.0, end: float | None = None,
                   include_camera: bool = True) -> str:
        end = self.duration if end is None else end
        return "\n".join(b.line() for b in self.beats_between(start, end)
                         if include_camera or b.kind != "CAMERA")


def parse_episode(path: Path) -> Episode:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    number, logline, notes = 0, "", ""
    scene_ends: list[float] = []
    beats: list[Beat] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("EPISODE NUMBER:"):
            m = re.search(r"(\d+)", line)
            number = int(m.group(1)) if m else 0
        elif line.startswith("EPISODE LOGLINE:"):
            logline = line.split(":", 1)[1].strip()
        elif line.startswith("CONTINUITY NOTES:"):
            notes = line.split(":", 1)[1].strip()
        elif (m := SCENE_TIME_RE.match(line)) and m.group(1) == "end":
            scene_ends.append(to_seconds(m.group(2), m.group(3)))
        elif (h := HEADER_RE.match(line)) and i + 1 < len(lines):
            t = TIME_RE.search(lines[i + 1])
            if t:
                # Text runs until the next blank line.
                body, j = [], i + 2
                while j < len(lines) and lines[j].strip():
                    body.append(lines[j].strip())
                    j += 1
                name, tone = h.group(1).strip(), (h.group(2) or "").strip()
                kind = name if name in NON_DIALOGUE else "DIALOGUE"
                text = " ".join(body).strip()
                if kind == "DIALOGUE":
                    text = text.strip('"').strip("“”")
                beats.append(Beat(
                    idx=0, kind=kind,
                    start=to_seconds(t.group(1), t.group(2)),
                    end=to_seconds(t.group(3), t.group(4)),
                    text=text,
                    speaker=name if kind == "DIALOGUE" else "",
                    tone=tone,
                ))
                i = j
                continue
        i += 1

    beats.sort(key=lambda b: (b.start, b.kind != "CAMERA"))
    for k, b in enumerate(beats):
        b.idx = k
    duration = max(scene_ends + [b.end for b in beats] + [0.0])
    if not number:
        m = re.search(r"episode-(\d+)", Path(path).name)
        number = int(m.group(1)) if m else 0
    return Episode(number, Path(path), logline, notes, duration, beats)


def load_show(folder: str | Path) -> list[Episode]:
    files = sorted(Path(folder).glob("*.txt"))
    episodes = [parse_episode(f) for f in files]
    episodes = [e for e in episodes if e.beats]
    return sorted(episodes, key=lambda e: e.number)

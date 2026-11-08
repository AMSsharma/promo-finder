#!/usr/bin/env python3
"""Find the best promo clips in a microdrama screenplay.

    python find_promos.py --input ../           # all episodes, auto-picked model
    python find_promos.py --input ../ --provider gemini --model gemini-2.5-flash
    python find_promos.py --input ../ --provider rules     # offline, no key

Pipeline
  1. parse    screenplay -> timed beats                       (code)
  2. propose  per-episode promo candidates + key moments      (LLM, 1 call/episode)
  3. clean    snap to shots, no mid-line cuts, 20-60s, flags  (code)
  4. stitch   multi-segment promos across episodes            (LLM, 1 call)
  5. judge    a "never seen the show" viewer rates top clips  (LLM, 1 call/clip)
  6. rank     weighted score + penalties + diversity          (code)
  7. report   promos.json / promos.csv / report.md / raw/     (code)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import prompts
import rules
from cutting import Promo, Segment, check, clean_segment, clip_script, fit_length
from llm import LLM, LLMError, load_dotenv, pick_provider
from screenplay import Episode, fmt, load_show, parse_time

HERE = Path(__file__).resolve().parent

# Ranking weights. Kept here so they are easy to argue with and change.
W_HOOK, W_CLIFF, W_STANDALONE = 0.40, 0.35, 0.25
PENALTY = {"too_short": 2.0, "too_long": 2.0, "hook_late": 1.5, "slow_open": 1.0,
           "soft_ending": 1.0, "many_characters": 0.5, "cuts_mid_line": 1.0}
# New viewers who click a promo land on Episode 1. A promo whose payoff is in
# the first episodes delivers on its promise soonest and spoils the least.
def funnel_bonus(first_ep: int) -> float:
    return 0.5 if first_ep <= 10 else 0.25 if first_ep <= 20 else 0.0


def parse_range(text: str | None, eps: list[Episode]) -> list[Episode]:
    if not text:
        return eps
    wanted: set[int] = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        wanted.update(range(int(a), int(b or a) + 1))
    return [e for e in eps if e.number in wanted]


def safe_time(value, default: float) -> float:
    try:
        return parse_time(str(value).split()[0])
    except (ValueError, IndexError):
        return default


# ---------------------------------------------------------------- stages
def propose(llm: LLM, ep: Episode, prev: list[Episode], title: str, k: int,
            raw_dir: Path) -> tuple[list[Promo], list[dict]]:
    story = " | ".join(f"Ep {p.number}: {p.logline}" for p in prev[-3:]) or "(first episode)"
    user = prompts.EPISODE_USER.format(
        title=title, story_so_far=story, number=ep.number, logline=ep.logline,
        notes=ep.notes or "-", duration=fmt(ep.duration), transcript=ep.transcript(), k=k)
    data = llm.json(prompts.EPISODE_SYSTEM, user)
    (raw_dir / f"ep_{ep.number:03d}.json").write_text(json.dumps(data, indent=2))

    promos = []
    for c in data.get("candidates", [])[:k]:
        start = safe_time(c.get("start"), -1)
        end = safe_time(c.get("end"), -1)
        if start < 0 or end <= start or start > ep.duration:
            continue
        notes: list[str] = []
        seg = clean_segment(ep, start, min(end, ep.duration), notes)
        p = Promo(
            id="", segments=[seg], source="single",
            hook=c.get("hook", ""), hook_type=c.get("hook_type", ""),
            cliffhanger=c.get("cliffhanger", ""), why=c.get("why", ""),
            context_needed=c.get("context_needed", ""), scores=c.get("scores", {}),
            hook_time=safe_time(c.get("hook_time"), start), adjustments=notes)
        if end > ep.duration + 1:
            p.adjustments.append(f"model end {c.get('end')} was past episode end")
        promos.append(p)
    moments = [dict(m, episode=ep.number) for m in data.get("moments", [])]
    return promos, moments


def stitch(llm: LLM, show: dict[int, Episode], moments: list[dict], title: str,
           n: int, raw_dir: Path) -> list[Promo]:
    # Keep this prompt small: free tiers cap tokens per request/minute.
    lines = "\n".join(f"Ep {e.number}: {e.logline[:110]}" for e in show.values())
    keep = [m for m in moments if m.get("kind") in ("hook", "cliffhanger")] or moments
    mom = "\n".join(f"Ep {m['episode']} | {m.get('start')}-{m.get('end')} | "
                    f"{m.get('kind', '')} | {str(m.get('desc', ''))[:90]}" for m in keep[:70])
    user = prompts.STITCH_USER.format(title=title, loglines=lines, moments=mom, n=n)
    data = llm.json(prompts.STITCH_SYSTEM, user, max_tokens=2500)
    (raw_dir / "stitch.json").write_text(json.dumps(data, indent=2))

    out = []
    for c in data.get("promos", [])[:n]:
        segs, notes = [], []
        for s in c.get("segments", []):
            ep = show.get(int(s.get("episode", -1)))
            a, b = safe_time(s.get("start"), -1), safe_time(s.get("end"), -1)
            if not ep or a < 0 or b <= a:
                notes.append(f"dropped invalid segment {s}")
                continue
            segs.append(clean_segment(ep, a, min(b, ep.duration), notes))
        if len(segs) < 1:
            continue
        out.append(Promo(
            id="", segments=segs, source="stitched" if len(segs) > 1 else "single",
            hook=c.get("hook", ""), hook_type=c.get("hook_type", ""),
            cliffhanger=c.get("cliffhanger", ""),
            why=(c.get("title", "") + ": " if c.get("title") else "") + c.get("why", ""),
            context_needed=c.get("context_needed", ""), scores=c.get("scores", {}),
            adjustments=notes))
    return out


def judge(llm: LLM, promo: Promo, show: dict[int, Episode], raw_dir: Path) -> None:
    user = prompts.JUDGE_USER.format(duration=promo.length, clip=clip_script(promo, show))
    try:
        data = llm.json(prompts.JUDGE_SYSTEM, user, max_tokens=1500)
    except LLMError as e:
        promo.flags.append(f"judge_failed ({e})")
        return
    (raw_dir / f"judge_{promo.id}.json").write_text(json.dumps(data, indent=2))
    promo.judge = {k: v for k, v in data.items() if k != "_raw"}


def num(x, default=5.0) -> float:
    try:
        return max(1.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return default


def score(p: Promo) -> None:
    hook, cliff, alone = (num(p.scores.get("hook")), num(p.scores.get("cliffhanger")),
                          num(p.scores.get("standalone")))
    if p.judge:   # an independent cold viewer pulls the self-scores toward reality
        hook = (hook + num(p.judge.get("first_3s_stop"))) / 2
        cliff = (cliff + num(p.judge.get("need_to_know_next"))) / 2
        alone = (alone + num(p.judge.get("understandable"))) / 2
    base = W_HOOK * hook + W_CLIFF * cliff + W_STANDALONE * alone
    pen = sum(v for f in p.flags for k, v in PENALTY.items() if f.startswith(k))
    bonus = funnel_bonus(min(p.episodes))
    p.score = round(base - pen + bonus, 2)
    p.score_parts = {"hook": round(hook, 1), "cliffhanger": round(cliff, 1),
                     "standalone": round(alone, 1), "penalty": -pen, "funnel_bonus": bonus}


def overlaps(a: Promo, b: Promo) -> bool:
    for x in a.segments:
        for y in b.segments:
            if x.episode == y.episode:
                inter = min(x.end, y.end) - max(x.start, y.start)
                if inter > 0.3 * min(x.length, y.length):
                    return True
    return False


def post_order(promos: list[Promo], n: int) -> list[Promo]:
    """Best first, skipping near-duplicates and capping 2 picks per episode."""
    chosen: list[Promo] = []
    per_ep: dict[int, int] = {}
    for p in sorted(promos, key=lambda p: p.score, reverse=True):
        if p.flags and any(f.startswith(("too_short", "too_long")) for f in p.flags):
            continue
        if any(overlaps(p, c) for c in chosen):
            continue
        if any(per_ep.get(e, 0) >= 2 for e in p.episodes):
            continue
        chosen.append(p)
        for e in p.episodes:
            per_ep[e] = per_ep.get(e, 0) + 1
        if len(chosen) >= n:
            break
    return chosen


# ---------------------------------------------------------------- reports
def promo_dict(p: Promo, rank: int | None) -> dict:
    return {
        "rank": rank, "id": p.id, "source": p.source, "score": p.score,
        "length_s": round(p.length, 1),
        "segments": [{"episode": s.episode, "start": fmt(s.start), "end": fmt(s.end)}
                     for s in p.segments],
        "hook": p.hook, "hook_type": p.hook_type, "cliffhanger": p.cliffhanger,
        "why": p.why, "context_needed": p.context_needed,
        "model_scores": p.scores, "score_parts": p.score_parts,
        "judge": p.judge, "flags": p.flags, "adjustments": p.adjustments,
    }


def write_reports(out: Path, promos: list[Promo], ranked: list[Promo], meta: dict,
                  show: dict[int, Episode]) -> None:
    rank_of = {p.id: i for i, p in enumerate(ranked, 1)}
    allp = sorted(promos, key=lambda p: (rank_of.get(p.id, 999), -p.score))
    (out / "promos.json").write_text(json.dumps(
        {"meta": meta, "promos": [promo_dict(p, rank_of.get(p.id)) for p in allp]}, indent=2))

    with open(out / "promos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "score", "length_s", "segments", "hook", "cliffhanger",
                    "why", "flags"])
        for p in allp:
            w.writerow([rank_of.get(p.id, ""), p.id, p.score, round(p.length, 1),
                        " + ".join(s.label() for s in p.segments), p.hook, p.cliffhanger,
                        p.why, "; ".join(p.flags)])

    md = [f"# Promo report: {meta['title']}", "",
          f"Model: `{meta['model']}` · episodes {meta['episodes']} · "
          f"{len(promos)} candidates · {meta['seconds']:.0f}s runtime · "
          f"{meta['llm_calls']} model calls ({meta['cache_hits']} cached) · "
          f"tokens in/out {meta['tokens']['input_tokens']}/{meta['tokens']['output_tokens']}",
          "", "Calls answered per model: " + (", ".join(
              f"`{k}` {v}" for k, v in meta.get("models_used", {}).items()) or "none"), "",
          "## Post these first", ""]
    for i, p in enumerate(ranked, 1):
        segs = " + ".join(s.label() for s in p.segments)
        md += [f"### {i}. {segs}  ({p.length:.0f}s, score {p.score})",
               f"- **Hook:** {p.hook}",
               f"- **Cliffhanger:** {p.cliffhanger}",
               f"- **Why it works:** {p.why}",
               f"- **Why this rank:** hook {p.score_parts['hook']}, cliffhanger "
               f"{p.score_parts['cliffhanger']}, standalone {p.score_parts['standalone']}"
               f", funnel bonus {p.score_parts['funnel_bonus']}, penalty {p.score_parts['penalty']}",
               ]
        if p.judge:
            md.append(f"- **Cold viewer:** \"{p.judge.get('what_happens', '')}\" · confusing: "
                      f"{p.judge.get('confusing', '')} · verdict **{p.judge.get('verdict', '')}**")
        if p.context_needed and p.context_needed.lower() != "none":
            md.append(f"- **Context a new viewer lacks:** {p.context_needed}")
        if p.flags:
            md.append(f"- **Flags:** {'; '.join(p.flags)}")
        md.append("")

    # quality section
    flagged = [p for p in promos if p.flags]
    counts: dict[str, int] = {}
    for p in promos:
        for f in p.flags:
            k = f.split(" ")[0]
            counts[k] = counts.get(k, 0) + 1
    snapped = sum(1 for p in promos if any("snapped" in a for a in p.adjustments))
    big_moves = [p for p in promos if any(_moved(a) > 3 for a in p.adjustments)]
    md += ["## Quality checks", "",
           f"- {len(flagged)}/{len(promos)} candidates raised at least one rule flag.",
           *[f"  - `{k}`: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])],
           f"- {snapped}/{len(promos)} candidates needed their cut points snapped to a shot "
           f"boundary; {len(big_moves)} moved more than 3s (the model's timestamps were off).",
           ]
    judged = [p for p in promos if p.judge]
    if judged:
        skip = [p for p in judged if str(p.judge.get("verdict", "")).lower() == "skip"]
        gap = [p for p in judged
               if num(p.scores.get("standalone")) - num(p.judge.get("understandable")) >= 3]
        md += [f"- Cold-viewer judge rated {len(judged)} promos: "
               f"{len(skip)} 'skip', and for {len(gap)} the proposer's standalone score was "
               f"3+ points higher than the cold viewer's (the proposer over-rates clips it "
               f"has context for)."]
        for p in gap[:5]:
            md.append(f"  - {p.id} ({' + '.join(s.label() for s in p.segments)}): "
                      f"proposer {p.scores.get('standalone')} vs viewer "
                      f"{p.judge.get('understandable')}: {p.judge.get('confusing', '')}")
    md += ["", "## All candidates", "",
           "| id | segments | len | score | flags |", "|---|---|---|---|---|"]
    for p in allp:
        md.append(f"| {p.id} | {' + '.join(s.label() for s in p.segments)} | {p.length:.0f}s | "
                  f"{p.score} | {'; '.join(p.flags)} |")
    (out / "report.md").write_text("\n".join(md) + "\n")


def models_in(raw: Path) -> dict:
    """Which model answered each saved raw response (cached ones included)."""
    counts: dict[str, int] = {}
    for f in sorted(raw.glob("*.json")):
        try:
            m = json.loads(f.read_text()).get("_model", "unknown")
        except (OSError, ValueError):
            m = "unreadable"
        counts[m] = counts.get(m, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _moved(note: str) -> float:
    try:
        a, b = note.split(": snapped ")[1].split(" -> ")
        (a1, a2), (b1, b2) = a.split("-"), b.split("-")
        return max(abs(parse_time(a1) - parse_time(b1)), abs(parse_time(a2) - parse_time(b2)))
    except (IndexError, ValueError):
        return 0.0


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="..", help="folder of screenplay .txt files")
    ap.add_argument("--title", default=None, help="show title (default: folder name)")
    ap.add_argument("--provider", default=None, help="gemini|groq|openai|anthropic|rules")
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", default=None, help="e.g. 1-10 or 1,4,7")
    ap.add_argument("--per-episode", type=int, default=3, help="candidates per episode")
    ap.add_argument("--stitch", type=int, default=5, help="cross-episode promos (0 = off)")
    ap.add_argument("--judge", type=int, default=15, help="cold-viewer checks on top N (0 = off)")
    ap.add_argument("--top", type=int, default=10, help="length of the 'post first' list")
    ap.add_argument("--fallback-models", default="",
                    help="comma list of models to switch to when one runs out of quota")
    ap.add_argument("--rpm", type=float, default=8, help="max model calls per minute (free tiers)")
    ap.add_argument("--out", default=None, help="output folder")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    load_dotenv(HERE / ".env")
    t0 = time.time()
    folder = Path(args.input).resolve()
    title = args.title or folder.name
    eps = load_show(folder)
    if not eps:
        print(f"no screenplays found in {folder}", file=sys.stderr)
        return 1
    show = {e.number: e for e in eps}
    chosen = parse_range(args.episodes, eps)

    provider = pick_provider(args.provider)
    llm = None
    if provider != "rules":
        llm = LLM(provider, args.model, None if args.no_cache else HERE / "outputs" / ".cache",
                  min_interval=60.0 / args.rpm if args.rpm else 0,
                  fallbacks=[m.strip() for m in args.fallback_models.split(",") if m.strip()])
    model_name = llm.name if llm else "rules (no model)"
    out = Path(args.out) if args.out else HERE / "outputs" / model_name.replace(":", "_").replace("/", "_")
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"{title}: {len(chosen)} episodes, model {model_name} -> {out}")

    promos: list[Promo] = []
    moments: list[dict] = []
    for ep in chosen:
        if llm:
            prev = [e for e in eps if e.number < ep.number]
            try:
                found, mom = propose(llm, ep, prev, title, args.per_episode, raw)
            except LLMError as e:
                print(f"  Ep {ep.number}: FAILED ({e}); falling back to rules")
                found, mom = rules.find(ep, args.per_episode), []
                for p in found:
                    p.flags.append("model_failed_used_rules")
            moments += mom
        else:
            found = rules.find(ep, args.per_episode)
        for j, p in enumerate(found, 1):
            p.id = f"E{ep.number:02d}-{j}"
        promos += found
        print(f"  Ep {ep.number}: {len(found)} candidates")

    if llm and args.stitch and moments:
        try:
            st = stitch(llm, show, moments, title, args.stitch, raw)
            for j, p in enumerate(st, 1):
                p.id = f"X-{j}"
            promos += st
            print(f"  stitched: {len(st)} cross-episode promos")
        except LLMError as e:
            print(f"  stitching FAILED ({e})")

    for p in promos:
        fit_length(p, show)
        check(p, show)
        score(p)

    if llm and args.judge:
        shortlist = sorted(promos, key=lambda p: p.score, reverse=True)[: args.judge]
        print(f"  judging top {len(shortlist)} with a cold-viewer prompt")
        for p in shortlist:
            judge(llm, p, show, raw)
            score(p)
        # Judging only lowers inflated self-scores, so an unjudged clip would
        # beat a judged one unfairly. Keep judging until the whole final list
        # has been seen by the cold viewer.
        for _ in range(12):
            todo = [p for p in post_order(promos, args.top) if not p.judge
                    and not any(f.startswith("judge_failed") for f in p.flags)]
            if not todo:
                break
            print(f"  judging {len(todo)} more that entered the top {args.top}")
            for p in todo:
                judge(llm, p, show, raw)
                score(p)

    ranked = post_order(promos, args.top)
    meta = {"title": title, "model": model_name, "episodes": args.episodes or "all",
            "seconds": time.time() - t0, "llm_calls": llm.calls if llm else 0,
            "cache_hits": llm.cached if llm else 0,
            "tokens": llm.usage if llm else {"input_tokens": 0, "output_tokens": 0},
            "models_used": models_in(raw),
            "settings": vars(args)}
    write_reports(out, promos, ranked, meta, show)
    print(f"done: {len(promos)} candidates, top {len(ranked)} in {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

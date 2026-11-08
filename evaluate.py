#!/usr/bin/env python3
"""Compare the tool's promos with an editor's hand-picked ones.

    python evaluate.py outputs/<run>/promos.json gold.json

gold.json: [{"episode": 1, "start": "00:09.80", "end": "00:50.40", "note": "..."}]

A gold pick counts as FOUND when some tool promo segment in the same episode
overlaps it by IoU >= 0.5. We also report how far the start point was off,
since a promo that starts 3s late loses its hook.
"""
import json
import sys

from screenplay import parse_time


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def main(pred_path, gold_path, top=None):
    promos = json.load(open(pred_path))["promos"]
    if top:
        promos = [p for p in promos if p.get("rank") and p["rank"] <= int(top)]
    gold = json.load(open(gold_path))
    found, rows = 0, []
    for g in gold:
        gs = (parse_time(g["start"]), parse_time(g["end"]))
        best = (0.0, None, None)
        for p in promos:
            for s in p["segments"]:
                if s["episode"] != g["episode"]:
                    continue
                ps = (parse_time(s["start"]), parse_time(s["end"]))
                v = iou(gs, ps)
                if v > best[0]:
                    best = (v, p["id"], ps[0] - gs[0])
        hit = best[0] >= 0.5
        found += hit
        rows.append((g, best, hit))
    for g, (v, pid, off), hit in rows:
        tag = "FOUND" if hit else "missed"
        extra = f"best {pid} IoU {v:.2f}, start off by {off:+.1f}s" if pid else "no overlap"
        print(f"{tag:6} Ep {g['episode']} {g['start']}-{g['end']}  ({extra})  {g.get('note', '')}")
    print(f"\nrecall: {found}/{len(gold)} editor picks found"
          + (f" in the top {top}" if top else " among all candidates"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])

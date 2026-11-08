# Promo Finder

Reads a microdrama's timed screenplays and finds 20–60 second promo clips that
open on a hook, end on a cliffhanger and make sense to someone new. It ranks them
in the order to post.

No installs needed: Python 3.9+ and the standard library only.

## Run it

```bash
cp .env.example .env          # add one free key (Gemini or Groq)
python3 find_promos.py --input ..                     # all episodes
python3 find_promos.py --input .. --episodes 1-10     # a subset
python3 find_promos.py --input .. --provider rules    # offline, no key
```

To swap the model, change one flag:

```bash
python3 find_promos.py --input .. --provider gemini --model gemini-2.5-flash
python3 find_promos.py --input .. --provider groq   --model openai/gpt-oss-120b
python3 find_promos.py --input .. --provider anthropic --model claude-opus-5-5
OPENAI_BASE_URL=https://your-gateway/v1 python3 find_promos.py --input .. --provider openai --model <name>
```

`--input` is any folder of screenplay `.txt` files in the Kuku format, so a new
show is just a new folder.

Score the tool against an editor's own picks:

```bash
python3 evaluate.py outputs/<run>/promos.json gold.json
```

## What comes out (`outputs/<provider>_<model>/`)

| File | What it is |
|---|---|
| `report.md` | Ranked "post these first" list, quality checks, all candidates |
| `promos.json` / `promos.csv` | Every promo: segments (episode, start, end), hook, cliffhanger, why, scores, flags |
| `raw/` | Unedited model responses: one per episode, the stitching call and each judge call |

## How it works

| Step | Who does it | Why |
|---|---|---|
| 1. Parse screenplay into timed beats (shot, action, sound, line) | code | exact and free |
| 2. Propose up to 3 promos and 2–4 "moments" per episode | **LLM**, 1 call per episode | judging drama needs language understanding |
| 3. Snap cuts to camera-shot boundaries, never cut mid-line, force 20–60s | code | models get timestamps slightly wrong; editors cut on shots |
| 4. Flag `hook_late`, `slow_open`, `soft_ending`, `many_characters`, `cuts_mid_line` | code (rules) | cheap, explainable checks |
| 5. Stitch 2–4 moments from different episodes into one promo | **LLM**, 1 call | premise-style ads ("jilted → rescued → ?") |
| 6. Cold-viewer judge: the model sees *only the clip* and rates it | **LLM**, 1 call per top clip | catches clips that only make sense with context |
| 7. Score = 0.40·hook + 0.35·cliffhanger + 0.25·standalone − penalties + early-episode bonus; dedupe; max 2 per episode | code | ranking rules you can argue with and change |

Model responses are cached in `outputs/.cache`, so a re-run is free, and
`--rpm` keeps calls under free-tier rate limits.

Free tiers cap requests per model per day. `--fallback-models` lists backups.
A backup can be from another provider, so a run can finish on Groq once
Gemini's quota is used up:

```bash
python3 find_promos.py --input .. --provider gemini --model gemini-3.5-flash \
  --fallback-models gemini-3.7-flash,gemini-flash-latest,groq:openai/gpt-oss-120b
```

Busy or rate-limited calls are retried, then rotated to the next model. A model
whose daily quota is gone is dropped for the rest of the run. `report.md` lists
how many calls each model answered.

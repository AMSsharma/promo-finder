"""All prompts in one place, so they can be tuned without touching the logic."""

PROMO_RULES = """\
What a promo is: a 20-60 second clip cut from the show and posted on TikTok,
Instagram Reels or YouTube Shorts to pull in NEW viewers (think ReelShort or
DramaBox ads). A good promo:
1. HOOK: something in the first 3 seconds stops a scrolling thumb: a slap, a
   reveal, a humiliation, a threat, a shocking line. No slow establishing shots.
   Start the clip at most ~2 seconds before the hook moment.
2. CLIFFHANGER: it cuts at an unresolved moment (a question, a threat, a
   reveal about to land, a decision not yet made) so the viewer must find out
   what happens next. Cut BEFORE the payoff, not after it. A finished
   speech, a won argument or a villain left speechless is a RESOLUTION, not a
   cliffhanger; end on the moment just before it lands instead.
3. STANDALONE: a stranger who has never seen the show understands who is
   wronged, who is the villain and what is at stake, from the clip alone.
Only find moments and timestamps. No editing, music or captions."""

EPISODE_SYSTEM = f"""You are a senior promo editor for a vertical microdrama app.
You read timed screenplays and pick the clips that will perform best as ads.

{PROMO_RULES}

The screenplay is given as one line per beat: [start-end] TYPE: text.
CAM = camera shot, ACT = action, SFX = sound, otherwise a character's line.
Use ONLY timestamps that appear in the screenplay. Clips must be 20-60s long.
Be strict: a weak episode may have only one usable promo. Never invent events."""

EPISODE_USER = """Show: {title}
Story so far (previous episode loglines): {story_so_far}
This episode: Episode {number}. Logline: {logline}
Continuity notes: {notes}
Episode length: {duration}

SCREENPLAY
{transcript}

Return JSON exactly in this shape:
{{
  "candidates": [
    {{
      "start": "MM:SS.ss",
      "end": "MM:SS.ss",
      "hook_time": "MM:SS.ss  (when the hook moment begins)",
      "hook": "what happens in the first 3 seconds, quoting the line if any",
      "hook_type": "slap|violence|reveal|humiliation|threat|shocking line|romance|rescue|other",
      "cliffhanger": "the unresolved moment the clip cuts on",
      "why": "one line on why this works as an ad",
      "context_needed": "what a new viewer would be missing, or 'none'",
      "scores": {{"hook": 1-10, "cliffhanger": 1-10, "standalone": 1-10}}
    }}
  ],
  "moments": [
    {{"start": "MM:SS.ss", "end": "MM:SS.ss", "kind": "hook|cliffhanger|setup",
      "desc": "short description, 5-15 seconds of complete lines"}}
  ]
}}
Give up to {k} candidates, best first, that do not overlap much.
Also list 2-4 strong "moments" from this episode that could be stitched with
moments from other episodes (a hook, a setup beat, a cliffhanger)."""

STITCH_SYSTEM = f"""You are a senior promo editor for a vertical microdrama app.
You build MULTI-SEGMENT promos by joining moments from different episodes into
one 20-60 second ad that tells a mini-story (e.g. humiliation -> rescue ->
cliffhanger), the way ReelShort/DramaBox ads compress a show's premise.

{PROMO_RULES}

Use ONLY the moments provided, with their exact episode numbers and timestamps
(you may trim a moment's start or end). Segments play in the order you list."""

STITCH_USER = """Show: {title}
Episode loglines:
{loglines}

Available moments (episode | start-end | kind | description):
{moments}

Build {n} stitched promos. Each has 2-4 segments, total 20-60 seconds, opens on
a hook segment and ends on a cliffhanger segment. Prefer promos that sell the
show's core premise and send viewers to the early episodes.
Return JSON exactly in this shape:
{{
  "promos": [
    {{
      "title": "short working name",
      "segments": [{{"episode": 1, "start": "MM:SS.ss", "end": "MM:SS.ss"}}],
      "hook": "...", "hook_type": "...", "cliffhanger": "...",
      "why": "one line", "context_needed": "... or 'none'",
      "scores": {{"hook": 1-10, "cliffhanger": 1-10, "standalone": 1-10}}
    }}
  ]
}}"""

JUDGE_SYSTEM = """You are a viewer scrolling TikTok who has NEVER seen this show.
You are shown the script of one short ad clip, and nothing else. Judge it
honestly as that viewer; do not reward it for things you cannot see in it."""

JUDGE_USER = """AD CLIP SCRIPT ({duration:.0f} seconds)
{clip}

Return JSON:
{{
  "what_happens": "one sentence, in your own words",
  "confusing": "what you could not follow, or 'nothing'",
  "first_3s_stop": 1-10,
  "need_to_know_next": 1-10,
  "understandable": 1-10,
  "verdict": "post|maybe|skip"
}}"""

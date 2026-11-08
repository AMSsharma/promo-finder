"""One small interface over several LLM providers.

Swap models with --provider / --model (or env vars). Uses only the Python
standard library so the tool runs anywhere with no installs.

Providers:
  gemini     Google AI Studio (free tier)            GEMINI_API_KEY
  groq       Groq (free tier, OpenAI-compatible)     GROQ_API_KEY
  openai     Any OpenAI-compatible endpoint           OPENAI_API_KEY, OPENAI_BASE_URL
             (OpenAI, OpenRouter, Ollama, an internal gateway...)
  anthropic  Anthropic Messages API                  ANTHROPIC_API_KEY
  rules      No model: offline keyword baseline (see rules.py)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5-5",
}
KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def pick_provider(requested: str | None) -> str:
    if requested:
        return requested
    if os.environ.get("PROMO_PROVIDER"):
        return os.environ["PROMO_PROVIDER"]
    for name in ("gemini", "groq", "anthropic", "openai"):
        if os.environ.get(KEY_ENV[name]):
            return name
    return "rules"


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, provider: str, model: str | None = None,
                 cache_dir: Path | None = None, min_interval: float = 0.0,
                 fallbacks: list[str] | None = None):
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"unknown provider {provider!r}")
        model = model or os.environ.get("PROMO_MODEL") or DEFAULT_MODELS[provider]
        # Free tiers cap requests per model per day. When one model's quota
        # runs out, move on to the next entry. Entries are "model" (same
        # provider) or "provider:model", so a Gemini run can finish on Groq.
        main = f"{provider}:{model}"
        chain = [main]
        for fb in fallbacks or []:
            entry = fb if ":" in fb else f"{provider}:{fb}"
            if entry not in chain:
                chain.append(entry)
        if not self._has_key(provider):
            raise LLMError(f"{KEY_ENV[provider]} is not set (put it in .env or the shell)")
        skipped = [c for c in chain if not self._has_key(c.split(":", 1)[0])]
        for c in skipped:
            print(f"    [no key for {c}: skipping it as a fallback]")
        self.chain = [c for c in chain if c not in skipped]
        self.primary = main
        self.cache_name = main          # stays fixed even if models rotate
        self.used_models: dict[str, int] = {}
        self._use(main)
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self._last_call = 0.0
        self.calls = 0
        self.cached = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @staticmethod
    def _has_key(provider: str) -> bool:
        if provider == "openai" and os.environ.get("OPENAI_BASE_URL"):
            return True
        return bool(os.environ.get(KEY_ENV.get(provider, ""), ""))

    def _use(self, entry: str) -> None:
        self.provider, self.model = entry.split(":", 1)
        self.key = os.environ.get(KEY_ENV[self.provider], "")

    # ---- public -------------------------------------------------------
    def json(self, system: str, user: str, max_tokens: int = 8000) -> dict:
        """Ask for a JSON object; returns the parsed dict (raw text kept under _raw)."""
        key = hashlib.sha256(
            f"{self.cache_name}\n{system}\n{user}".encode()).hexdigest()[:24]
        cache_file = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache_file and cache_file.exists():
            self.cached += 1
            return json.loads(cache_file.read_text())

        last_err = None
        self._use(self.primary)            # always try the main model first
        busy_strikes = 0
        for attempt in range(4 + 2 * len(self.chain)):
            try:
                raw = self._call(system, user, max_tokens)
                data = extract_json(raw)
                data["_raw"] = raw
                data["_model"] = self.name
                self.used_models[self.name] = self.used_models.get(self.name, 0) + 1
                if cache_file:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(data, indent=1))
                return data
            except (LLMError, json.JSONDecodeError, ValueError) as e:
                last_err = e
                msg = str(e)
                daily = "HTTP 429" in msg and ("PerDay" in msg or "per day" in msg.lower())
                if "HTTP 404" in msg or daily:
                    self._drop_model(msg)          # gone for today: never retry it
                elif "HTTP 413" in msg and len(self.chain) > 1:
                    self._rotate()                 # request too big for this tier: try another model
                elif "HTTP 429" in msg or "HTTP 503" in msg:
                    busy_strikes += 1              # short-term limit or overload
                    if busy_strikes % 2 == 0 and len(self.chain) > 1:
                        self._rotate()
                    else:
                        time.sleep(30)
                else:
                    time.sleep(3)
                if not self.chain:
                    raise LLMError(f"all models out of quota: {msg[:200]}")
        raise LLMError(f"{self.name} failed after retries: {last_err}")

    def _drop_model(self, why: str) -> None:
        bad = self.name
        self.chain = [m for m in self.chain if m != bad]
        print(f"    [{bad} out of daily quota or missing -> removed]")
        if self.chain:
            if bad == self.primary:
                self.primary = self.chain[0]
            self._use(self.chain[0])

    def _rotate(self) -> None:
        i = self.chain.index(self.name) if self.name in self.chain else -1
        self._use(self.chain[(i + 1) % len(self.chain)])
        print(f"    [busy -> trying {self.name}]")

    # ---- providers ----------------------------------------------------
    def _call(self, system: str, user: str, max_tokens: int) -> str:
        gap = time.time() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_call = time.time()
        self.calls += 1
        if self.provider == "gemini":
            return self._gemini(system, user, max_tokens)
        if self.provider == "anthropic":
            return self._anthropic(system, user, max_tokens)
        return self._openai_compat(system, user, max_tokens)

    def _gemini(self, system, user, max_tokens):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            # Gemini 2.5 "thinking" tokens count toward maxOutputTokens, so leave headroom.
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max(max_tokens, 12000),
                                 "responseMimeType": "application/json"},
        }
        data = post(url, body, {"x-goog-api-key": self.key})
        meta = data.get("usageMetadata", {})
        self.usage["input_tokens"] += meta.get("promptTokenCount", 0)
        self.usage["output_tokens"] += meta.get("candidatesTokenCount", 0)
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise LLMError(f"gemini returned no text: {str(data)[:300]}")

    def _openai_compat(self, system, user, max_tokens):
        base = {"groq": "https://api.groq.com/openai/v1"}.get(
            self.provider, os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        body = {
            "model": self.model, "temperature": 0.2, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        data = post(base.rstrip("/") + "/chat/completions", body, headers)
        u = data.get("usage", {})
        self.usage["input_tokens"] += u.get("prompt_tokens", 0)
        self.usage["output_tokens"] += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]

    def _anthropic(self, system, user, max_tokens):
        body = {
            "model": self.model, "max_tokens": max(max_tokens, 16000),
            "system": system + "\nReply with a single JSON object and nothing else.",
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": os.environ.get("PROMO_EFFORT", "medium")},
        }
        headers = {"x-api-key": self.key, "anthropic-version": "2026-10-21"}
        if self.model.startswith(("claude-opus-5", "claude-fable-5", "claude-sonnet-5-5")):
            # On a safety-classifier decline, let the API retry on a fallback model.
            body["fallbacks"] = "default"
            headers["anthropic-beta"] = "server-side-fallback-2026-07-01"
        data = post("https://api.anthropic.com/v1/messages", body, headers, timeout=600)
        if data.get("stop_reason") == "refusal":
            raise LLMError("anthropic refused the request")
        u = data.get("usage", {})
        self.usage["input_tokens"] += u.get("input_tokens", 0)
        self.usage["output_tokens"] += u.get("output_tokens", 0)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def post(url: str, body: dict, headers: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        # Some gateways (e.g. Groq's) reject Python's default User-Agent.
        headers={"Content-Type": "application/json", "User-Agent": "promo-finder/1.0", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code}: {e.read()[:4000].decode(errors='replace')}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # Treat timeouts like "busy" so the caller waits, retries or rotates.
        raise LLMError(f"HTTP 503 (network/timeout): {e}")


def extract_json(text: str) -> dict:
    """Parse a JSON object even if the model wrapped it in prose or ``` fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a == -1 or b <= a:
            raise
        return json.loads(text[a:b + 1])

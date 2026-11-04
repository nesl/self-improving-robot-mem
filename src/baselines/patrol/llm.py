"""LLM agents on a patrol bank, against the local vLLM server.

    python3 -m baselines.patrol.llm --bank banks/hh_s0_p4.jsonl --out llm/ \
        --memory naive recent summary --told told not_told --look on off

Three memories, each in a told and a not-told arm, each with the free
look on or off:

* ``naive``: the object's sighting history (newest 60) and the spots
  checked empty since its last sighting. The old llm_belief floor, in
  the plain prompt format of this study.
* ``recent``: the object's last 10 sightings plus the most recent patrol
  listing of every room. Nothing older.
* ``summary``: once per day the model writes short notes (each resident's
  routine, each moving object's usual spots) from that day's movements
  plus the previous notes; questions are answered from the notes plus
  the object's last 3 sightings.

The told arm sees the residents' dated messages (bank header
``hint_messages``) from each shift day's first question on. Every agent
sees the identical patrol stream the classical agents see. Prompts are
checked for hidden state before they are sent (``leak_check``); a leak
aborts the run. Completions are cached by prompt hash, so a rerun with
the same bank and arm replays byte-identically without the server.
Temperature 0, seeded. A completion that does not parse falls back to
the last sighting with confidence 0 and is counted (``fallback``).
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from baselines.bank import JsonlBank
from baselines.patrol.bank import card_sentences
from baselines.patrol.leak_check import check_prompt
from baselines.types import DAY_SECONDS, ON_PERSON, OUT_OF_HOUSE, Episode, Observation, SenseResult

# OpenAI-compatible chat-completions server (the paper used vLLM serving Qwen3.8-27B). Set PATROL_LLM_ENDPOINT
# to your server; the endpoint is not part of the response-cache key.
ENDPOINT = os.environ.get("PATROL_LLM_ENDPOINT", "http://localhost:8000")
MODEL = os.environ.get("PATROL_LLM_MODEL", "Qwen/Qwen3.8-27B")
MEMORIES = ("naive", "recent", "summary", "routine", "routine7", "retrieval", "longcontext", "reflect")
RETRIEVAL_RECENT = 8            # retrieval: most recent sightings shown regardless of time of day
RETRIEVAL_TOD_WINDOW_S = 5400   # retrieval: "same time of day" = within 90 minutes of the query's time-of-day
RETRIEVAL_TOD_MAX = 10          # retrieval: cap on same-time-of-day sightings shown (oldest dropped first)
ABSTAIN = "ABSTAIN"
WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

ANSWER_SCHEMA_LOOK: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "confidence": {"type": "number"},
        "look_room": {"type": ["string", "null"]},
    },
    "required": ["ranking", "confidence", "look_room"], "additionalProperties": False}
ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "confidence": {"type": "number"},
    },
    "required": ["ranking", "confidence"], "additionalProperties": False}
CONF_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        # 600 until 2026-09-25, raised for the same reason the chooser's 240 went: a reasoning
        # field is not a memory field and must not bind. The ORDER here was always right -
        # reasoning before the location - which is what made the chooser's order visible as a
        # fault rather than a convention.
        "reasoning": {"type": "string", "maxLength": 1200},
        "location": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["reasoning", "location", "confidence"], "additionalProperties": False}
CONF_SCALE = ("Confidence is the probability, from 0 to 1, that the spot you name is right. Low numbers are "
              "expected when the evidence is thin: about 0.3 means a guess among several plausible spots, "
              "about 0.6 means it is the most likely spot but it could easily be elsewhere, about 0.9 means "
              "you have seen it there repeatedly at this time of day and nothing has changed. "
              "Do not give the same number every time.")


# ------------------------------------------------------------------ client --

class LLMClient:
    """Chat completions against vLLM with a file cache keyed by prompt hash.
    Local server: the spend is zero dollars; tokens are still counted."""

    def __init__(self, cache_dir: pathlib.Path, endpoint: str = ENDPOINT, model: str = MODEL,
                 replay_only: bool = False) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint, self.model, self.replay_only = endpoint, model, replay_only
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    ATTEMPTS = 2            # one retry
    # RAISED from 900 to 2400 on 2026-09-25, and this is infrastructure rather than a
    # setting of the study. The deadline is not in the cache key and does not reach the
    # server: at temperature 0 and seed 0 a night that takes 1,100 seconds and succeeds
    # writes exactly what the same night would have written in 700. All the deadline
    # decides is whether a night is LOST, and a lost night is not neutral - the notes
    # carry forward unchanged, so the arm it happens to is handicapped for that night.
    #
    # Why 900 was too small, in the same units for once. `max_tokens` is
    # `300 + 260 * the night's edit allowance`, the allowance is derived from what the
    # night saw and has no upper bound, and the largest so far is 97 edits, which asks
    # for 25,520 tokens. Measured on this server under ten concurrent cells, generation
    # runs at 27 to 33 tokens a second. So 900 seconds buys about 27,000 tokens: the
    # biggest night we have already run was inside the wall by less than 6%, and one
    # busier household crosses it. 2,400 seconds covers 30,000 tokens even if contention
    # halves the rate to 20 a second.
    #
    # The cost, stated because it is the only one: a request that is genuinely hung now
    # holds its slot for up to 2,400 seconds per attempt instead of 900, and ATTEMPTS is
    # 2, so a hung night can occupy a slot for 80 minutes. Losing 80 minutes of one slot
    # is cheaper than losing a night of one arm's memory.
    #
    # The NEXT limit in the chain, so nobody has to rediscover it: the server is started
    # with --max-model-len 98304, and a request is prompt tokens plus `max_tokens`. The
    # worst so far is about 57,000 - a 31,000-token prompt plus 25,520 - so there is
    # headroom, but the prompt grows with the notes and this is the limit to watch after
    # the clock.
    DEADLINE_S = 2400.0     # hard wall-clock cap per attempt, enforced by a watchdog that kills the socket

    def key(self, messages: List[dict], schema: Optional[dict], max_tokens: int) -> str:
        blob = json.dumps({"model": self.model, "messages": messages, "schema": schema,
                           "max_tokens": max_tokens, "temperature": 0, "seed": 0}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _post(self, body: dict) -> dict:
        """One POST to /v1/chat/completions with a HARD deadline. urllib's socket timeout alone did not fire on a
        request vLLM lost (2026-09-22 00:27: one arm's thread sat in poll() for an hour, no exception, blocking the
        pool's shutdown and so the whole chain) — so a watchdog thread shuts the socket down at DEADLINE_S, which
        makes the blocked recv raise and lets the retry/fallback path run instead of main() hanging."""
        u = urllib.parse.urlsplit(self.endpoint)
        # The socket timeout is the DEADLINE, not a smaller number. It used to be
        # `min(600.0, self.DEADLINE_S)` with DEADLINE_S at 900, so the socket gave up three
        # hundred seconds before the watchdog that exists to bound this call - two limits on
        # one thing, and the smaller one was the real one and nobody chose it.
        #
        # Measured on 2026-09-25: every one of 24 failed calls in an overnight run was
        # `TimeoutError: timed out` at exactly 600 seconds, on both attempts, from arms whose
        # nightly generation is long. The calls were not lost, they were still working. Five
        # cells were held back by a gate because of it. The watchdog at DEADLINE_S still
        # bounds the call, so nothing here waits longer than it ever could - a request that is
        # genuinely lost is still killed at 900 seconds.
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=self.DEADLINE_S)
        done = threading.Event()

        def watchdog():
            if done.is_set():
                return
            try:
                if conn.sock is not None:
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

        timer = threading.Timer(self.DEADLINE_S, watchdog)
        timer.daemon = True
        timer.start()
        try:
            conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {data[:200]!r}")
            return json.loads(data)
        finally:
            done.set()
            timer.cancel()
            try:
                conn.close()
            except OSError:
                pass

    def complete(self, messages: List[dict], schema: Optional[dict], max_tokens: int) -> Tuple[Optional[str], dict]:
        key = self.key(messages, schema, max_tokens)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            with self.lock:
                self.stats["cached"] += 1
            return rec["text"], rec.get("usage", {})
        if self.replay_only:
            return None, {}
        body: Dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                "temperature": 0, "seed": 0,
                                "chat_template_kwargs": {"enable_thinking": False}}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        t0 = time.time()
        text, usage = None, {}
        for attempt in range(self.ATTEMPTS):
            try:
                d = self._post(body)
                text = d["choices"][0]["message"]["content"]
                usage = d.get("usage") or {}
                break
            except Exception as e:  # noqa: BLE001 - server hiccup or lost request: retry once, then give up (fallback answer)
                print(f"llm call failed ({attempt + 1}/{self.ATTEMPTS}, {time.time() - t0:.0f}s in): {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                with self.lock:
                    self.stats["failed_attempts"] = self.stats.get("failed_attempts", 0) + 1
                time.sleep(2 * (attempt + 1))
        if text is None:
            with self.lock:
                self.stats["lost"] = self.stats.get("lost", 0) + 1
            print(f"llm call LOST after {self.ATTEMPTS} attempts ({time.time() - t0:.0f}s): falling back for this question",
                  file=sys.stderr, flush=True)
        dt = time.time() - t0
        with self.lock:
            self.stats["calls"] += 1
            self.stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            self.stats["completion_tokens"] += int(usage.get("completion_tokens", 0))
            self.stats["seconds"] += dt
        if text is not None:
            # per-thread temp name: two arms can ask the identical prompt at once
            tmp = path.with_name(f"{path.stem}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"text": text, "usage": usage, "model": self.model}))
            try:
                tmp.replace(path)
            except OSError:
                tmp.unlink(missing_ok=True)
        return text, usage


class AnthropicClient(LLMClient):
    """The same interface on the Anthropic Messages API (added 2026-11-02 for the camera-ready pilot).

    Same file cache and key as LLMClient; the model name is in the key, so Qwen and Claude entries
    never mix. What differs from the vLLM call, because claude-sonnet-5-5 rejects the alternatives:
    no temperature or seed (non-default sampling values return a 400), thinking off is
    `{"type": "between_tools"}` (`disabled` returns a 400), and forced tool use returns a 400, so the
    schema fallback is a strict tool with tool_choice auto. The schema is enforced with
    output_config.format; if the API rejects it, keywords it does not support (maxLength and the
    like) are removed and checked here instead, and only if that is still rejected is the tool used.
    Every real call appends its token counts to `usage_log` (or $PATROL_LLM_USAGE_LOG)."""

    UNSUPPORTED = ("maxLength", "minLength", "minimum", "maximum", "exclusiveMinimum",
                   "exclusiveMaximum", "multipleOf", "maxItems", "minItems", "pattern")
    PREFIX_MIN_CHARS = 4000     # shorter shared prefixes are below the API's minimum cacheable size

    def __init__(self, cache_dir: pathlib.Path, model: str = "claude-sonnet-5-5",
                 replay_only: bool = False, usage_log: Optional[pathlib.Path] = None,
                 max_retries: int = 8) -> None:
        super().__init__(cache_dir, endpoint="anthropic", model=model, replay_only=replay_only)
        import anthropic
        self.anthropic = anthropic
        # The SDK retries 408/409/429/5xx (529 overloaded included) with exponential backoff.
        self.api = None if replay_only else anthropic.Anthropic(max_retries=max_retries, timeout=self.DEADLINE_S)
        log = usage_log or os.environ.get("PATROL_LLM_USAGE_LOG")
        self.usage_log = pathlib.Path(log) if log else None
        self.recent: Dict[str, List[str]] = defaultdict(list)
        self.start_at: Dict[str, str] = {}    # per schema: "sanitized" or "tool" after a rejection
        self.stats.update({"cache_read_tokens": 0, "cache_write_tokens": 0, "unparsable": 0,
                           "schema_sanitized": 0, "tool_fallback": 0, "refusals": 0})

    @classmethod
    def _sanitize(cls, schema: Any) -> Any:
        if isinstance(schema, dict):
            t = schema.get("type")
            if isinstance(t, list) and "null" in t and "enum" in schema:
                # the API rejects an enum on a type list: same values, as anyOf
                rest = {k: v for k, v in schema.items() if k not in ("type", "enum")}
                values = [v for v in schema["enum"] if v is not None]
                return {"anyOf": [{"type": [x for x in t if x != "null"][0], "enum": values},
                                  {"type": "null"}], **cls._sanitize(rest)}
            out = {k: cls._sanitize(v) for k, v in schema.items() if k not in cls.UNSUPPORTED}
            if isinstance(schema.get("required"), list):
                # draft 2020-12 wants unique entries; vLLM took a doubled "why" (the told-if-right
                # edits schema), the API rejects it at every level, so ACE's nightly edit was lost
                out["required"] = list(dict.fromkeys(schema["required"]))
            return out
        if isinstance(schema, list):
            return [cls._sanitize(v) for v in schema]
        return schema

    @classmethod
    def _clip(cls, schema: Any, value: Any) -> Tuple[Any, int]:
        """Cut what constrained decoding would have stopped: strings past maxLength and lists past
        maxItems, both dropped by _sanitize. Qwen's grammar cannot exceed them, so cutting (not
        rejecting the whole reply) is the like-for-like treatment. Returns (value, number of cuts)."""
        if not isinstance(schema, dict):
            return value, 0
        if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
            return value[:schema["maxLength"]], 1
        if isinstance(value, list):
            n = 0
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                value, n = value[:schema["maxItems"]], 1
            out = []
            for v in value:
                v, k = cls._clip(schema.get("items") or {}, v)
                out.append(v); n += k
            return out, n
        if isinstance(value, dict):
            n = 0
            for k, sub in (schema.get("properties") or {}).items():
                if k in value:
                    value[k], c = cls._clip(sub, value[k]); n += c
            return value, n
        return value, 0

    @classmethod
    def _violations(cls, schema: Any, value: Any) -> List[str]:
        """What cannot be cut into shape after _clip: a list with fewer than minItems items
        (e.g. the nightly "at least one edit" Qwen's grammar enforced)."""
        out: List[str] = []
        if isinstance(schema, dict):
            if isinstance(value, list):
                if "minItems" in schema and len(value) < schema["minItems"]:
                    out.append(f"fewer than {schema['minItems']} items")
                for v in value:
                    out += cls._violations(schema.get("items") or {}, v)
            if isinstance(value, dict):
                for k, sub in (schema.get("properties") or {}).items():
                    if k in value:
                        out += cls._violations(sub, value[k])
        return out

    def _user_blocks(self, system: str, user: str) -> List[dict]:
        """Split the user text at the longest line-aligned prefix it shares with a recent prompt
        under the same system prompt, and mark that prefix for caching (the household header)."""
        best = 0
        for other in self.recent[system]:
            n = 0
            for a, b in zip(user, other):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        with self.lock:
            self.recent[system] = (self.recent[system] + [user])[-8:]
        cut = user.rfind("\n", 0, best) + 1
        if cut < self.PREFIX_MIN_CHARS:
            return [{"type": "text", "text": user}]
        return [{"type": "text", "text": user[:cut], "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": user[cut:]}]

    def _request(self, system: str, user: str, schema: Optional[dict], max_tokens: int,
                 how: str) -> Tuple[Optional[str], Any]:
        kw: Dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens,
            "thinking": {"type": "between_tools"},
            "messages": [{"role": "user", "content": self._user_blocks(system, user)}]}
        if system:
            kw["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if schema is not None and how in ("format", "sanitized"):
            sc = schema if how == "format" else self._sanitize(schema)
            kw["output_config"] = {"format": {"type": "json_schema", "schema": sc}}
        elif schema is not None:
            tool = {"name": "answer", "description": "Give your reply by calling this tool.",
                    "strict": True, "input_schema": self._sanitize(schema)}
            kw["tools"], kw["tool_choice"] = [tool], {"type": "auto"}
            kw["messages"][0]["content"].append(
                {"type": "text", "text": "Reply by calling the answer tool."})
        with self.api.messages.stream(**kw) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            with self.lock:
                self.stats["refusals"] += 1
            return None, msg
        if how == "tool":
            use = next((b for b in msg.content if b.type == "tool_use"), None)
            return (json.dumps(use.input) if use is not None else None), msg
        return "".join(b.text for b in msg.content if b.type == "text"), msg

    API_TOKEN_FACTOR = 2

    REWORDINGS = (("your own thinking", "your own words"), ("think about", "consider"),
                  (", from your own reading of it", ", as judged earlier tonight"),
                  ("What you said you wanted changed tonight", "Changes suggested earlier tonight"),
                  # the room-choice prompt (field "why"): this line alone triggers it (2/2 refused; reworded 2/2 through)
                  ("Keep your reasoning under", "Keep your why under"),
                  # from a bisection of a differently worded nightly prompt (4 nights x 2 tries: without
                  # these 8/8 refused, with them 0/8); must come after "your own thinking" -> "your own words"
                  ("and why it happened. What you write down is your own words, not a copy of what you saw.",
                   "and what caused it. What you write down is your conclusions, not a copy of what you saw."),
                  ("if something you believe turned out to be wrong.", "if one of your notes turned out to be wrong."))

    @classmethod
    def _reword(cls, system: str, user: str, schema: Optional[dict]) -> Tuple[str, str, Optional[dict], bool]:
        """The Sonnet-only retry after a refusal: field "reasoning" -> "why", and two phrases reworded."""
        for a, b in cls.REWORDINGS:
            system, user = system.replace(a, b), user.replace(a, b)
        props = (schema or {}).get("properties") or {}
        if "reasoning" not in props or "why" in props:
            return system, user, schema, False
        sc = json.loads(json.dumps(schema))
        sc["properties"] = {("why" if k == "reasoning" else k): v for k, v in sc["properties"].items()}
        sc["required"] = ["why" if k == "reasoning" else k for k in sc.get("required", [])]
        return system, user, sc, True

    def complete(self, messages: List[dict], schema: Optional[dict], max_tokens: int) -> Tuple[Optional[str], dict]:
        key = self.key(messages, schema, max_tokens)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            with self.lock:
                self.stats["cached"] += 1
            return rec["text"], rec.get("usage", {})
        if self.replay_only:
            return None, {}
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        t0 = time.time()
        schema_id = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
        text, usage, how, err = None, {}, self.start_at.get(schema_id, "format"), None
        sent_schema, reworded, renamed = schema, False, False
        # SONNET-ONLY: the API is given API_TOKEN_FACTOR x the caller's max_tokens. The sanitised schema has no
        # maxLength, so Sonnet writes longer fields than Qwen's grammar allowed and ran out (ACE judge-the-day
        # at 1200, 2026-11-02). The fields are then cut to their limits by _clip. The cache key keeps the
        # caller's max_tokens, so replies cached before this change still hit.
        api_max_tokens = int(max_tokens * self.API_TOKEN_FACTOR)
        while True:
            try:
                text, msg = self._request(system, user, sent_schema, api_max_tokens, how)
                u = msg.usage
                usage = {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
                         "cache_read_input_tokens": u.cache_read_input_tokens or 0,
                         "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
                         "stop_reason": msg.stop_reason, "how": how}
                if msg.stop_reason == "refusal":   # which classifier fired, and why
                    rdir = os.environ.get("PATROL_LLM_REFUSAL_DIR")
                    if rdir:   # the prompt itself: stop_details came back empty on every refusal seen so far
                        pathlib.Path(rdir).mkdir(parents=True, exist_ok=True)
                        (pathlib.Path(rdir) / f"{key}.json").write_text(json.dumps(
                            {"system": system, "user": user, "schema": schema, "how": how}, indent=1))
                    sd = getattr(msg, "stop_details", None)
                    if sd is not None:
                        sd = sd.model_dump() if hasattr(sd, "model_dump") else sd
                        usage["refusal"] = {"category": (sd or {}).get("category") if isinstance(sd, dict) else str(sd),
                                            "explanation": str((sd or {}).get("explanation", ""))[:300] if isinstance(sd, dict) else ""}
                    if not reworded:
                        # SONNET-ONLY, and only after a refusal (calls that went through are sent as before).
                        # Measured 2026-11-02 on captured prompts, 2 tries each: the answer step is refused
                        # because its JSON field is named "reasoning" (renamed to "why": 2/2 through; same
                        # field without a schema, or with the length sentence removed: 2/2 refused). The
                        # nightly prompt goes through with "your own thinking"/"think about" reworded (2/2), and the
                        # ACE night-1 prompt also needed its "from your own reading of it" header reworded (2/2).
                        # The reply is mapped back, so callers see the field names they asked for.
                        reworded = True
                        system, user, sent_schema, renamed = self._reword(system, user, sent_schema)
                        with self.lock:
                            self.stats["reworded_after_refusal"] = self.stats.get("reworded_after_refusal", 0) + 1
                        continue
                if renamed and text is not None:
                    try:
                        obj = json.loads(text)
                        if isinstance(obj, dict) and "why" in obj and "reasoning" not in obj:
                            obj["reasoning"] = obj.pop("why")
                            text = json.dumps(obj)
                    except ValueError:
                        pass
                usage["reworded_after_refusal"] = reworded
                break
            except self.anthropic.BadRequestError as e:
                # a schema the structured-output feature will not take: sanitise, then the tool
                if schema is not None and how != "tool" and "schema" in str(e).lower():
                    how = "sanitized" if how == "format" else "tool"
                    self.start_at[schema_id] = how
                    with self.lock:
                        self.stats["schema_sanitized" if how == "sanitized" else "tool_fallback"] += 1
                    continue
                err = e
                break
            except self.anthropic.APIError as e:   # retries exhausted, or a connection that never came back
                err = e
                break
        if err is not None:
            print(f"anthropic call failed: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        if text is not None and schema is not None:
            try:
                obj, cuts = self._clip(schema, json.loads(text))
                if cuts:
                    text = json.dumps(obj)
                    usage["cuts_to_schema_limits"] = cuts
                    with self.lock:
                        self.stats["cuts_to_schema_limits"] = self.stats.get("cuts_to_schema_limits", 0) + cuts
                bad = self._violations(schema, obj)
            except ValueError:
                bad = ["not JSON"]
            if bad and any("fewer than" in b for b in bad) and err is None:
                # SONNET-ONLY: one retry that states the item-count rule the sanitised schema could
                # not enforce. Logged as retried_for_item_count so it can be disclosed.
                rule = "; ".join(sorted(set(b for b in bad if "items" in b)))
                try:
                    text, msg = self._request(system, user + f"\n\nYour reply must obey this rule, which a reply "
                                              f"just broke: no list may have {rule}. For the edits list, that "
                                              f"means at least one edit.", sent_schema, api_max_tokens, how)
                    for k_api, k_us in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                                        ("cache_read_input_tokens", "cache_read_input_tokens"),
                                        ("cache_creation_input_tokens", "cache_creation_input_tokens")):
                        usage[k_us] = usage.get(k_us, 0) + (getattr(msg.usage, k_api, 0) or 0)
                    usage["retried_for_item_count"] = True
                    with self.lock:
                        self.stats["retried_for_item_count"] = self.stats.get("retried_for_item_count", 0) + 1
                    if renamed and text is not None:
                        obj = json.loads(text)
                        if isinstance(obj, dict) and "why" in obj and "reasoning" not in obj:
                            obj["reasoning"] = obj.pop("why"); text = json.dumps(obj)
                    if text is not None:
                        obj, cuts = self._clip(schema, json.loads(text))
                        text = json.dumps(obj)
                        usage["cuts_to_schema_limits"] = usage.get("cuts_to_schema_limits", 0) + cuts
                        bad = self._violations(schema, obj)
                    else:
                        bad = ["no reply"]
                except (ValueError, self.anthropic.APIError) as e:
                    bad = [f"retry failed: {type(e).__name__}"]
            if bad:
                with self.lock:
                    self.stats["unparsable"] += 1
                print(f"anthropic reply rejected: {bad}", file=sys.stderr, flush=True)
                usage["rejected_for"] = bad[:5]
                text = None
        dt = time.time() - t0
        with self.lock:
            self.stats["calls"] += 1
            if text is None:
                self.stats["lost"] = self.stats.get("lost", 0) + 1
            self.stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            self.stats["completion_tokens"] += int(usage.get("completion_tokens", 0))
            self.stats["cache_read_tokens"] += int(usage.get("cache_read_input_tokens", 0))
            self.stats["cache_write_tokens"] += int(usage.get("cache_creation_input_tokens", 0))
            self.stats["seconds"] += dt
            if self.usage_log is not None:
                with self.usage_log.open("a") as fh:
                    fh.write(json.dumps({"key": key, "seconds": round(dt, 2), "ok": text is not None,
                                         # which call this was: answer (reasoning/location), choice (why/room), night (edits)
                                         "schema_fields": sorted(((schema or {}).get("properties") or {}).keys()),
                                         "error": None if err is None else f"{type(err).__name__}: {err}"[:300],
                                         **usage}) + "\n")
        if text is not None:
            tmp = path.with_name(f"{path.stem}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"text": text, "usage": usage, "model": self.model}))
            try:
                tmp.replace(path)
            except OSError:
                tmp.unlink(missing_ok=True)
        return text, usage


def make_client(cache_dir: pathlib.Path, backend: Optional[str] = None, **kw: Any) -> LLMClient:
    """`--backend anthropic` or PATROL_LLM_BACKEND=anthropic picks AnthropicClient; vLLM otherwise."""
    backend = backend or os.environ.get("PATROL_LLM_BACKEND", "vllm")
    if backend == "anthropic":
        return AnthropicClient(cache_dir, **kw)
    return LLMClient(cache_dir, **kw)


# ------------------------------------------------------------------ memory --

FLAT_WEEK = False
"""Set per bank in run_arm: True when the bank's calendar runs the weekday routine on every day (regime banks with
``day_kind: weekday`` in every stage). Then no weekday name is shown anywhere - timestamps are "day 18 10:40",
resident cards drop their weekend sentence and messages are dated by day index - so the model is not told it is
Saturday in a world that is not keeping Saturdays."""


def flat_card(card: dict) -> str:
    """card_sentences without the weekend sentence and with 'Weekdays:' read as every day."""
    s = (f"{card['name']} (in their {card['age_band']}) {card['occupation']}. "
         f"{card['weekday'].replace('Weekdays:', 'Every day:')}")
    if card["hobbies"]:
        s += f" Hobbies: {', '.join(card['hobbies'])}."
    if card.get("pet"):
        s += f" {card['name']} {card['pet']}."
    return s


def day_label(day: int, day_names: Dict[int, str]) -> str:
    return f"day {day}" if FLAT_WEEK else day_names.get(day, f"day {day}")


def hhmm(t: int) -> str:
    hh, mm = divmod((int(t) % DAY_SECONDS) // 60, 60)
    return f"{hh:02d}:{mm:02d}"


def clock(t: int, day_names: Dict[int, str]) -> str:
    """'day 18 Fri 10:40': the day index is part of every timestamp, so a
    sighting from an earlier week cannot be mistaken for one from today
    (a multi-week study has four or five of every weekday)."""
    day = int(t) // DAY_SECONDS
    if FLAT_WEEK:
        return f"day {day} {hhmm(t)}"
    return f"day {day} {day_names.get(day, f'day {day}')[:3]} {hhmm(t)}"


class Memory:
    """What an LLM agent has seen: the same stream the classical beliefs
    get, kept in the forms the three prompts read from."""

    def __init__(self, rooms: Dict[str, List[str]], rec_room: Dict[str, str], names: Dict[str, str]) -> None:
        self.rooms, self.rec_room, self.names = rooms, rec_room, names
        self.sightings: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        self.empties: Dict[str, Dict[str, int]] = defaultdict(dict)      # obj -> rec -> newest empty time
        self.snapshot: Dict[str, Dict[str, Any]] = {}                    # room -> latest listing
        self.moves: Dict[int, List[str]] = defaultdict(list)             # day -> plain-word movement lines
        self.presence: Dict[int, List[str]] = defaultdict(list)          # day -> who was home per pass
        self.classes: Dict[str, str] = {}

    def last_spot(self, obj: str) -> Optional[str]:
        s = self.sightings.get(obj)
        return s[-1][1] if s else None

    def update_group(self, t: int, items: Sequence[Any], day_names: Dict[int, str], is_look: bool = False) -> None:
        """One instant's evidence: observations and/or per-spot listings."""
        listed_rooms = set()
        seen_now: Dict[str, str] = {}
        residents_now: List[str] = []
        for it in items:
            if isinstance(it, Observation):
                seen_now[it.object_id] = it.receptacle_id
                self.classes.setdefault(it.object_id, it.object_class)
                continue
            room = self.rec_room.get(it.receptacle_id)
            if room is None:
                continue
            listed_rooms.add(room)
            snap = self.snapshot.get(room)
            if snap is None or snap["t"] != t:
                snap = {"t": t, "contents": {}, "residents": []}
                self.snapshot[room] = snap
            snap["contents"][it.receptacle_id] = list(it.contents)
            for res in it.residents_present:
                nm = self.names.get(res, res)
                if nm not in snap["residents"]:
                    snap["residents"].append(nm)
                if nm not in residents_now:
                    residents_now.append(nm)
            for o in it.contents:
                seen_now[o] = it.receptacle_id
                self.classes.setdefault(o, it.object_classes.get(o, ""))
        day = t // DAY_SECONDS
        walkthrough = not self.sightings         # the walkthrough itself is not a movement
        for o, rec in sorted(seen_now.items()):
            prev = self.last_spot(o)
            if prev != rec and not walkthrough:
                self.moves[day].append(f"{clock(t, day_names)} {o}: {prev or 'not seen before'} -> {rec}")
            self.sightings[o].append((t, rec))
        # empties: every known object absent from a listed spot
        for it in items:
            if isinstance(it, SenseResult):
                for o in self.sightings:
                    if o not in it.contents:
                        self.empties[o][it.receptacle_id] = t
        full_pass = listed_rooms and listed_rooms >= set(self.rooms)
        if full_pass and not is_look:
            self.presence[day].append(f"{clock(t, day_names)}: {', '.join(residents_now) or 'nobody'}")
            for o in sorted(self.sightings):
                prev = self.last_spot(o)
                if o not in seen_now and prev not in (None, "gone"):
                    self.moves[day].append(f"{clock(t, day_names)} {o}: {prev} -> not in any room")
                    self.sightings[o].append((t, "gone"))   # marker: last full pass did not find it

    def history(self, obj: str) -> List[Tuple[int, str]]:
        return [(t, r) for t, r in self.sightings.get(obj, []) if r != "gone"]

    def checked_since_last_sighting(self, obj: str) -> List[Tuple[int, str]]:
        """(t, spot) listings after the object's last sighting that did not
        show it (the newest per spot)."""
        h = self.history(obj)
        t_last = h[-1][0] if h else -1
        return sorted((t, r) for r, t in self.empties.get(obj, {}).items() if t > t_last)

    def checked_lines(self, obj: str, day_names: Dict[int, str], limit: int = 12) -> List[str]:
        """The same, one line per instant: a full patrol that found nothing
        is one line, a whole room is one item, single spots are listed."""
        by_t: Dict[int, List[str]] = defaultdict(list)
        for t, r in self.checked_since_last_sighting(obj):
            by_t[t].append(r)
        all_spots = {r for recs in self.rooms.values() for r in recs}
        lines = []
        for t in sorted(by_t):
            spots = set(by_t[t])
            if spots >= all_spots:
                lines.append(f"- {clock(t, day_names)}: full patrol, not found in any room")
                continue
            parts = []
            for room, recs in sorted(self.rooms.items()):
                if set(recs) <= spots:
                    parts.append(f"{room} (all spots)")
                else:
                    parts += [r for r in recs if r in spots]
            lines.append(f"- {clock(t, day_names)}: " + ", ".join(parts))
        return lines[-limit:]

    def last_spot_real(self, obj: str) -> Optional[str]:
        h = self.history(obj)
        return h[-1][1] if h else None


# ----------------------------------------------------------------- prompts --

SYSTEM = ("You help a home robot keep track of where household things are. "
          "Read the robot's notes and answer with JSON only.")
SYSTEM_NOTES = ("You help a home robot keep track of where household things are. "
                "You keep short plain-text notes about the home; write the notes as plain text, not JSON.")


def rooms_block(rooms: Dict[str, List[str]]) -> List[str]:
    return [f"- {room}: {', '.join(recs)}" for room, recs in sorted(rooms.items())]


def patrol_words(patrol_hours: int, patrol_times: Optional[List[str]] = None) -> str:
    if patrol_times:
        if len(patrol_times) == 1:
            return f"once a day at {patrol_times[0]}"
        return "every day at " + ", ".join(patrol_times[:-1]) + f" and {patrol_times[-1]}"
    return f"every {patrol_hours} hour{'s' if patrol_hours != 1 else ''}"


def header_lines(t: int, day_names: Dict[int, str], cards: List[dict], rooms: Dict[str, List[str]],
                 patrol_hours: int, look_on: bool, hints: List[str], fmt: str = "rank",
                 patrol_times: Optional[List[str]] = None, question_moments: str = "",
                 feedback_delay_min: Optional[float] = None) -> List[str]:
    day = t // DAY_SECONDS
    # The static part (protocol, residents, rooms, instructions) comes first and the time-varying "Now:" line
    # last, so every prompt of a household shares a long prefix and the server's prefix cache serves the prefill.
    L = [f"The robot walked through every room on {day_label(0, day_names)} at 18:00 and since then patrols "
         f"every room {patrol_words(patrol_hours, patrol_times)}, listing what is on every spot "
         f"and who is in the room. Pockets and bags are not looked into."]
    if look_on:
        L.append("Before answering this question the robot may look at ONE room right now, for free.")
    L += ["", "Residents:"] + [f"- {flat_card(c) if FLAT_WEEK else card_sentences(c)}" for c in cards]
    L += ["", "Rooms and spots:"] + rooms_block(rooms)
    if fmt == "conf" and question_moments == "activity":
        from baselines.llm_hypotheses.protocol_text import QUESTIONS_ACTIVITY
        L += ["", QUESTIONS_ACTIVITY.replace(" on a weekday and on a weekend", "") if FLAT_WEEK else QUESTIONS_ACTIVITY]
    if fmt == "conf" and feedback_delay_min is not None:
        L += ["", ("At its nightly round the robot is told, for each of the day's questions, where the object turned "
                   "out to be; those show up in the sightings below like any other sighting." if feedback_delay_min < 0 else
                   f"About {feedback_delay_min:g} minutes after each question the resident tells the robot where the "
                   f"object turned out to be; those show up in the sightings below like any other sighting.")]
    if fmt == "conf":
        L += ["", "Answer with one spot name from the list above (the object is somewhere in the house; if it "
                  "is being used or carried right now, name the spot it is most likely to be at or next to)."]
    else:
        L += ["", f"Answer options: a spot name from the list above; {ON_PERSON}:<name> if a resident is carrying it; "
                  f"{OUT_OF_HOUSE} if it has been taken out of the home; {ABSTAIN} if you would rather not guess."]
    if fmt == "conf":
        L += ["", CONF_SCALE, "",
              'Reply with JSON: {"reasoning": one or two short sentences on what the sightings, the time of day and '
              'the residents\' routine suggest, "location": one spot name, "confidence": number from 0 to 1}']
    if hints:
        L += ["", "Messages from the residents:"] + [f"- {h}" for h in hints]
    L += ["", f"Now: {day_label(day, day_names)}, {hhmm(t)} (day {day} of the study; "
              f"the walkthrough was on {day_label(0, day_names)} evening)."]
    return L


def _tod_gap(t1: int, t2: int) -> int:
    """Seconds between two timestamps' time-of-day, taking the shorter way around the clock."""
    d = abs((t1 % DAY_SECONDS) - (t2 % DAY_SECONDS))
    return min(d, DAY_SECONDS - d)


def memory_lines(memory: Memory, mem_kind: str, obj: str, day_names: Dict[int, str], notes: Optional[str],
                 query_t: Optional[int] = None) -> List[str]:
    h = memory.history(obj)
    L: List[str] = []
    if mem_kind == "naive":
        shown = h[-60:]
        L.append(f"Sightings of {obj} (oldest first{'; newest 60 of ' + str(len(h)) if len(h) > 60 else ''}):")
        L += [f"- {clock(t, day_names)}: {r}" for t, r in shown] or ["- none"]
        L += ["", "Listings since the last sighting that did NOT show it:"]
        L += memory.checked_lines(obj, day_names, 40) or ["- none"]
    elif mem_kind == "recent":
        L.append(f"Last sightings of {obj} (oldest first, at most 10):")
        L += [f"- {clock(t, day_names)}: {r}" for t, r in h[-10:]] or ["- none"]
        L += ["", "Most recent patrol listing of each room:"]
        for room in sorted(memory.rooms):
            snap = memory.snapshot.get(room)
            if snap is None:
                L.append(f"- {room}: not listed yet")
                continue
            who = f"; present: {', '.join(snap['residents'])}" if snap["residents"] else "; nobody present"
            spots = "; ".join(f"{rec}: {', '.join(objs) if objs else 'empty'}" for rec, objs in sorted(snap["contents"].items()))
            L.append(f"- {room} ({clock(snap['t'], day_names)}{who}): {spots}")
    elif mem_kind in ("summary", "routine", "routine7"):
        L.append("Your notes on this home (written at the end of each day so far):")
        L.append(notes.strip() if notes else "(no notes yet: only the walkthrough has happened)")
        L += ["", f"Last sightings of {obj} (at most 3):"]
        L += [f"- {clock(t, day_names)}: {r}" for t, r in h[-3:]] or ["- none"]
        chk = memory.checked_lines(obj, day_names, 6)
        if chk:
            L += ["", "Listings since the last sighting that did NOT show it:"] + chk
    elif mem_kind == "reflect":
        # Reflexion-style (Shinn et al. 2023): the notes are not a summary of what happened, only of what the
        # model got WRONG and why (written nightly from that day's found-it feedback) — distinct from routine7's
        # full sighting-table rewrite, which keeps everything, right or wrong.
        L.append("Notes on mistakes you have made so far (written each night from that day's corrections):")
        L.append(notes.strip() if notes else "(no notes yet: no mistakes recorded)")
        L += ["", f"Last sightings of {obj} (at most 10):"]
        L += [f"- {clock(t, day_names)}: {r}" for t, r in h[-10:]] or ["- none"]
        chk = memory.checked_lines(obj, day_names, 6)
        if chk:
            L += ["", "Listings since the last sighting that did NOT show it:"] + chk
    elif mem_kind == "retrieval":
        # Mem0 / A-Mem style (Chhikara et al. 2025; Xu et al. 2025): pull what is relevant to THIS query instead of
        # dumping the whole history — relevance here is recency and same-time-of-day (our sightings are structured
        # (time, receptacle) tuples, not free text, so time-of-day stands in for a semantic-similarity retrieval key).
        recent = h[-RETRIEVAL_RECENT:]
        recent_ts = {t for t, _ in recent}
        if query_t is not None:
            tod = [(t, r) for t, r in h if t not in recent_ts and _tod_gap(t, query_t) <= RETRIEVAL_TOD_WINDOW_S]
            tod = tod[-RETRIEVAL_TOD_MAX:]
        else:
            tod = []
        L.append(f"Retrieved sightings of {obj} — most recent (oldest first, at most {RETRIEVAL_RECENT}):")
        L += [f"- {clock(t, day_names)}: {r}" for t, r in recent] or ["- none"]
        L += ["", f"Retrieved sightings of {obj} at about this time of day on other days (oldest first, at most {RETRIEVAL_TOD_MAX}):"]
        L += [f"- {clock(t, day_names)}: {r}" for t, r in tod] or ["- none"]
        chk = memory.checked_lines(obj, day_names, 10)
        if chk:
            L += ["", "Listings since the last sighting that did NOT show it:"] + chk
    elif mem_kind == "longcontext":
        # the control nobody has run: the model's whole context is not curated at all — every sighting of this
        # object AND the household's whole movement log (every object), unclipped; measures whether scale alone
        # substitutes for a memory architecture (cf. "Lost in the Middle", Liu et al. 2024, on whether long-context
        # models actually use what is given them).
        L.append(f"Every sighting of {obj} so far (oldest first, {len(h)} total):")
        L += [f"- {clock(t, day_names)}: {r}" for t, r in h] or ["- none"]
        L += ["", "Listings since the last sighting that did NOT show it:"]
        L += memory.checked_lines(obj, day_names, 200) or ["- none"]
        L += ["", "Every movement the household's patrol has noticed, any object (oldest first):"]
        all_moves = [m for day in sorted(memory.moves) for m in memory.moves[day]]
        L += [f"- {m}" for m in all_moves] or ["- none"]
    else:
        raise ValueError(mem_kind)
    return L


def question_messages(memory: Memory, mem_kind: str, q, day_names, cards, rooms, patrol_hours, look_on, hints,
                      notes, look_result: Optional[str], fmt: str = "rank",
                      patrol_times: Optional[List[str]] = None, question_moments: str = "",
                      feedback_delay_min: Optional[float] = None) -> List[dict]:
    L = header_lines(q.t_query, day_names, cards, rooms, patrol_hours, look_on and look_result is None, hints, fmt, patrol_times,
                     question_moments, feedback_delay_min)
    L += ["", f"Question: where is {q.object_id} (a {q.object_class.replace('_', ' ')}) right now?", ""]
    L += memory_lines(memory, mem_kind, q.object_id, day_names, notes, query_t=q.t_query)
    if fmt == "conf":
        pass   # the confidence scale and the reply format are in the static header (prefix-cache friendly)
    elif look_result is not None:
        L += ["", look_result]
        L += ["", 'Reply with JSON: {"ranking": [up to 3 answer options, most likely first], '
                  '"confidence": probability from 0 to 1 that the first one is right}']
    elif look_on:
        L += ["", 'Reply with JSON: {"ranking": [up to 3 answer options, most likely first], '
                  '"confidence": probability from 0 to 1 that the first one is right, '
                  '"look_room": a room name to look at right now before you answer, or null to answer without looking}']
    else:
        L += ["", 'Reply with JSON: {"ranking": [up to 3 answer options, most likely first], '
                  '"confidence": probability from 0 to 1 that the first one is right}']
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(L)}]


def notes_messages(memory: Memory, day: int, day_names, cards, rooms, patrol_hours, hints, notes: Optional[str],
                   walkthrough: Optional[str]) -> List[dict]:
    t_end = (day + 1) * DAY_SECONDS - 60
    L = header_lines(t_end, day_names, cards, rooms, patrol_hours, False, hints)
    L += ["", f"It is the end of {day_label(day, day_names)}. Update your notes on this home.", ""]
    L += ["Your previous notes:", notes.strip() if notes else "(none yet)", ""]
    if walkthrough:
        L += ["What the walkthrough found:", walkthrough, ""]
    L.append(f"Movements the patrol noticed on {day_label(day, day_names)} (an object listed on a new spot, "
             f"or not found in any room):")
    L += [f"- {m}" for m in memory.moves.get(day, [])] or ["- none"]
    L += ["", "Who was home at each patrol:"]
    L += [f"- {p}" for p in memory.presence.get(day, [])] or ["- no full patrol yet"]
    L += ["", "Write the updated notes in plain text, at most 350 words: each resident's routine as you now "
              "understand it (when they are out, where they spend time" + (", by weekday and weekend" if not FLAT_WEEK else "") + "), and for each "
              "object that moves, where it usually is by time of day and what tends to take it out of the house. "
              "Keep what is still true from the previous notes, drop what turned out wrong."]
    return [{"role": "system", "content": SYSTEM_NOTES}, {"role": "user", "content": "\n".join(L)}]


def routine_messages(memory: Memory, day: int, day_names, cards, rooms, patrol_hours, hints, notes: Optional[str],
                     window_days: int = 1) -> List[dict]:
    """Nightly consolidation into a per-object routine table: the day's sightings grouped by object (movers only,
    i.e. objects seen at more than one spot so far), and the model rewrites the table of where each mover is by
    time of day. Same information as the ``summary`` notes, presented per object instead of as a movement list."""
    t_end = (day + 1) * DAY_SECONDS - 60
    L = header_lines(t_end, day_names, cards, rooms, patrol_hours, False, hints)
    L += ["", f"It is the end of {day_label(day, day_names)}. Update your routine table for this home.", ""]
    L += ["Your previous routine table:", notes.strip() if notes else "(none yet)", ""]
    d0, d1 = max(0, day - window_days + 1) * DAY_SECONDS, (day + 1) * DAY_SECONDS
    span = "Today's sightings" if window_days == 1 else f"Sightings over the last {window_days} days (one line per object and day)"
    L.append(f"{span}, per object (only objects that have been seen at more than one spot so far; "
             f"time: spot, in order; the 03:00 entry is the nightly round):")
    n = 0
    for o in sorted(memory.sightings):
        h = memory.history(o)
        if len({r for _, r in h}) < 2:
            continue
        for dd in range(max(0, day - window_days + 1), day + 1):
            today = [(t, r) for t, r in h if dd * DAY_SECONDS <= t < (dd + 1) * DAY_SECONDS]
            if not today:
                continue
            n += 1
            tag = "" if window_days == 1 else (f" day {dd}:" if FLAT_WEEK else f" day {dd} {day_names.get(dd, '')[:3]}:")
            L.append(f"- {o} ({memory.classes.get(o, '')}):{tag} " + ", ".join(f"{hhmm(t)} {r}" for t, r in today))
    if n == 0:
        L.append("- none")
    L += ["", "Rewrite the routine table in plain text, at most 600 words. One line per object that moves, in the form "
              "'object: <spot> <time window>; <spot> <time window>; ...' giving where it is through a typical day "
              "(when it is in use as well as where it rests), and note it when " + ("" if FLAT_WEEK else "weekend days differ or when ")
              + "the routine has changed recently (say since when). Objects that never move need no line. Keep what is "
              "still true from the previous table, correct what " + ("these" if window_days > 1 else "today's") + " sightings "
              "contradict, and drop nothing that they did not contradict."]
    return [{"role": "system", "content": SYSTEM_NOTES}, {"role": "user", "content": "\n".join(L)}]


def reflect_messages(memory: Memory, day: int, day_records: List[dict], day_names, cards, rooms, patrol_hours,
                     hints, notes: Optional[str]) -> List[dict]:
    """Reflexion-style (Shinn et al. 2023): notes are written from what the model got WRONG today and the
    found-it feedback that corrected it, not from the sightings themselves — distinct from a routine-table
    rewrite, which keeps a full record of right and wrong sightings alike."""
    t_end = (day + 1) * DAY_SECONDS - 60
    L = header_lines(t_end, day_names, cards, rooms, patrol_hours, False, hints)
    L += ["", f"It is the end of {day_label(day, day_names)}. You answered some questions wrong today; the resident's "
              "found-it feedback told you the truth 10 minutes after each one. Update your notes on your mistakes.", ""]
    L += ["Your previous mistake notes:", notes.strip() if notes else "(none yet)", ""]
    wrong = [r for r in day_records if not r.get("correct")]
    if wrong:
        L.append(f"Today's wrong answers ({len(wrong)} of {len(day_records)} questions), with what the feedback showed:")
        for r in wrong:
            L.append(f"- {clock(r['t_query'], day_names)}: asked where {r['object_id']} was; you said {r['answer']}; "
                     f"it was actually at {r['truth']}.")
    else:
        L.append("Every question today was answered correctly.")
    L += ["", "Write updated mistake notes in plain text, at most 250 words: for each object or situation you keep "
              "getting wrong, one line on the pattern (what you assumed, what was actually true, what to check "
              "instead next time). Keep entries that are still relevant, drop ones today's evidence corrected, add "
              "new ones only for real, repeated misses — a single one-off is not a pattern."]
    return [{"role": "system", "content": SYSTEM_NOTES}, {"role": "user", "content": "\n".join(L)}]


NOTICING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean"},
        "who": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
    },
    "required": ["changed", "who", "confidence"], "additionalProperties": False}


def noticing_messages(memory: Memory, day: int, day_names, cards, rooms, patrol_hours, hints) -> List[dict]:
    """Once a day: a direct self-report, independent of mem_kind and asked in every arm — the coordinator's
    'does the LLM notice' number, scored as a detector (fire days per household, false alarms) against the
    change alarm run on the classical counters."""
    t = (day + 1) * DAY_SECONDS - 30
    L = header_lines(t, day_names, cards, rooms, patrol_hours, False, hints)
    names = ", ".join(c["name"] for c in cards)
    L += ["", f"It is the end of {day_label(day, day_names)}. Based only on what the patrol and the found-it "
              f"feedback have shown you so far, has any resident's daily routine changed in the last day or two "
              f"— things showing up in different places than their usual pattern, for one person and not the "
              f"others? Residents: {names}.", "",
          'Reply with JSON: {"changed": true or false, "who": the resident\'s name if changed else null, '
          '"confidence": probability from 0 to 1 that you are right}']
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(L)}]


# ------------------------------------------------------------------ parsing --

def parse_answer(text: Optional[str], allowed: set, rooms: set) -> Tuple[Optional[List[str]], Optional[float], Optional[str], Optional[str], str]:
    """(ranking normalized to bank tokens, confidence, look_room, on_person_resident, status)."""
    if not text:
        return None, None, None, None, "no_json"
    try:
        obj = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        try:
            obj = json.loads(text[start:end + 1])
        except ValueError:
            return None, None, None, None, "no_json"
    if not isinstance(obj, dict) or not isinstance(obj.get("ranking"), list) or not obj["ranking"]:
        return None, None, None, None, "bad_shape"
    conf = obj.get("confidence")
    conf = min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) else None
    look = obj.get("look_room")
    look = look if isinstance(look, str) and look in rooms else None
    names: List[str] = []
    carrier = None
    for name in obj["ranking"]:
        if not isinstance(name, str):
            return None, conf, look, None, "bad_shape"
        name = name.strip()
        up = name.upper()
        if up.startswith(ON_PERSON):
            carrier = carrier or name[len(ON_PERSON):].strip(" :()") or None
            name = ON_PERSON
        elif up.startswith(OUT_OF_HOUSE):
            name = OUT_OF_HOUSE
        elif up == ABSTAIN:
            name = ABSTAIN
        if name not in names:
            names.append(name)
    if names[0] not in allowed and names[0] != ABSTAIN:
        return None, conf, look, carrier, "off_list"
    return names, conf, look, carrier, "ok"


def parse_conf(text: Optional[str], allowed: set) -> Tuple[Optional[str], Optional[float], str, str]:
    """(location normalized, confidence, reasoning, status) for the confidence format."""
    if not text:
        return None, None, "", "no_json"
    try:
        obj = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        try:
            obj = json.loads(text[start:end + 1])
        except ValueError:
            return None, None, "", "no_json"
    if not isinstance(obj, dict) or not isinstance(obj.get("location"), str):
        return None, None, "", "bad_shape"
    conf = obj.get("confidence")
    conf = min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) else None
    why = obj.get("reasoning") if isinstance(obj.get("reasoning"), str) else ""
    loc = obj["location"].strip()
    if loc not in allowed:
        low = {a.lower(): a for a in allowed}
        loc = low.get(loc.lower(), loc)
    if loc not in allowed:
        return None, conf, why, "off_list"
    return loc, conf, why, "ok"


# --------------------------------------------------------------------- run --

def run_arm(bank_path: pathlib.Path, mem_kind: str, told: bool, look_on: bool, client: LLMClient,
            out_dir: pathlib.Path, max_days: Optional[int] = None, fmt: str = "rank") -> List[dict]:
    """``fmt`` ``rank``: the overnight format (ranking, confidence, optional
    look); ``conf``: one in-house spot plus a confidence, no look, no
    ON_PERSON / OUT_OF_HOUSE / ABSTAIN."""
    header = json.loads(bank_path.read_text().splitlines()[0])
    global FLAT_WEEK
    kinds = set((header.get("day_kinds") or {}).values())
    FLAT_WEEK = bool(kinds) and kinds == {"weekday"}
    patrol_times = header.get("patrol_times") or None
    question_moments = header.get("protocol", {}).get("question_moments", "")
    feedback_delay_min = header.get("protocol", {}).get("feedback_delay_min")
    if fmt == "conf":
        look_on = False
    episode: Episode = next(iter(JsonlBank(bank_path).episodes()))
    day_names = {int(k): v for k, v in header["day_names"].items()}
    cards = header["protocol"]["residents"]
    names = {c["resident_id"]: c["name"] for c in cards}
    patrol_hours = int(header["patrol_hours"])
    hint_rows = header.get("hint_messages", []) if told else []
    if FLAT_WEEK:   # "Saturday: Yuki is home sick today." -> "Day 19: Yuki is home sick today."
        hint_rows = [{**h, "text": f"Day {h['day_index']}: " + h["text"].split(": ", 1)[-1]} for h in hint_rows]
    rooms: Dict[str, List[str]] = defaultdict(list)
    for rec, room in sorted(episode.receptacle_rooms.items()):
        if rec not in (ON_PERSON, OUT_OF_HOUSE):
            rooms[room].append(rec)
    rooms = {r: sorted(v) for r, v in rooms.items()}
    rec_room = {rec: room for room, recs in rooms.items() for rec in recs}
    allowed = set(episode.receptacle_ids)
    if fmt == "conf":
        allowed = {r for r in allowed if r not in (ON_PERSON, OUT_OF_HOUSE)}
    arm = f"llm_{mem_kind}/{'told' if told else 'not_told'}/look_{'on' if look_on else 'off'}"
    tag = {"household": header["household_id"], "patrol_hours": patrol_hours,
           "look": "llm" if look_on else "off", "agent": arm, "belief": arm, "memory": mem_kind, "told": told}
    label = header.get("patrol_label", f"p{patrol_hours}")
    run_dir = out_dir / f"{header['household_id']}_{label}_{mem_kind}_{'told' if told else 'nottold'}_look{'on' if look_on else 'off'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    calls = open(run_dir / "calls.jsonl", "w")

    def ask(messages: List[dict], schema: Optional[dict], max_tokens: int, where: str, llm_authored=()) -> Tuple[Optional[str], dict]:
        check_prompt(messages, where, llm_authored)
        text, usage = client.complete(messages, schema, max_tokens)
        calls.write(json.dumps({"where": where, "messages": messages, "completion": text, "usage": usage,
                                "llm_authored": [s for s in llm_authored if s]}) + "\n")
        calls.flush()
        return text, usage

    memory = Memory(rooms, rec_room, names)
    # walkthrough: initial sightings plus the empty-spot listings at tour_t
    tour_t = int(header["tour_t"])
    evidence = list(episode.evidence_stream())
    cursor = 0
    memory.update_group(tour_t, list(episode.initial_observations) + [e for e in evidence if e.t == tour_t], day_names)
    while cursor < len(evidence) and evidence[cursor].t <= tour_t:
        cursor += 1
    walkthrough_text = "\n".join(
        f"- {room}: " + "; ".join(f"{rec}: {', '.join(objs) if objs else 'empty'}"
                                  for rec, objs in sorted(memory.snapshot[room]["contents"].items()))
        for room in sorted(memory.snapshot)) if memory.snapshot else None
    # the walkthrough listing came in as sightings + empties; rebuild the full listing for the notes prompt
    tour_contents: Dict[str, Dict[str, List[str]]] = {room: {rec: [] for rec in recs} for room, recs in rooms.items()}
    for o in episode.initial_observations:
        if o.receptacle_id in rec_room:
            tour_contents[rec_room[o.receptacle_id]][o.receptacle_id].append(o.object_id)
    walkthrough_text = "\n".join(
        f"- {room}: " + "; ".join(f"{rec}: {', '.join(sorted(objs)) if objs else 'empty'}" for rec, objs in sorted(tour_contents[room].items()))
        for room in sorted(tour_contents))

    notes: Optional[str] = None
    notes_day = -1
    noticing_day = -1
    noticing: List[dict] = []
    records: List[dict] = []
    n_fallback = 0
    for day_questions in episode.questions_by_day:
        for q in day_questions:
            if max_days is not None and q.day_index > max_days:
                break
            # deliver the stream up to the question, one instant at a time
            while cursor < len(evidence) and evidence[cursor].t <= q.t_query:
                t = evidence[cursor].t
                grp = []
                while cursor < len(evidence) and evidence[cursor].t == t:
                    grp.append(evidence[cursor]); cursor += 1
                memory.update_group(t, grp, day_names)
            hints = [h["text"] for h in hint_rows if h["day_index"] <= q.day_index]
            # once a day, every arm regardless of mem_kind: a direct self-report ("has the routine changed?"),
            # scored later as a detector against the change alarm's fire days
            while noticing_day < q.day_index - 1:
                noticing_day += 1
                day_hints = [h["text"] for h in hint_rows if h["day_index"] <= noticing_day]
                nmsgs = noticing_messages(memory, noticing_day, day_names, cards, rooms, patrol_hours, day_hints)
                ntext, _ = ask(nmsgs, NOTICING_SCHEMA, 200, f"noticing day {noticing_day}")
                try:
                    nobj = json.loads(ntext) if ntext else {}
                except ValueError:
                    nobj = {}
                noticing.append({"day": noticing_day, "changed": bool(nobj.get("changed")), "who": nobj.get("who"),
                                 "confidence": nobj.get("confidence")})
            # daily notes: written at the first question of each day for the day before
            if mem_kind in ("summary", "routine", "routine7", "reflect"):
                while notes_day < q.day_index - 1:
                    notes_day += 1
                    day_hints = [h["text"] for h in hint_rows if h["day_index"] <= notes_day]
                    if mem_kind in ("routine", "routine7"):
                        msgs = routine_messages(memory, notes_day, day_names, cards, rooms, patrol_hours, day_hints, notes,
                                                window_days=7 if mem_kind == "routine7" else 1)
                    elif mem_kind == "reflect":
                        day_records = [r for r in records if r["day_index"] == notes_day]
                        msgs = reflect_messages(memory, notes_day, day_records, day_names, cards, rooms, patrol_hours, day_hints, notes)
                    else:
                        msgs = notes_messages(memory, notes_day, day_names, cards, rooms, patrol_hours, day_hints, notes,
                                              walkthrough_text if notes_day == 0 else None)
                    text, _ = ask(msgs, None, 1500 if mem_kind.startswith("routine") else 900, f"notes day {notes_day}", llm_authored=(notes,))
                    if text and text.strip():
                        notes = text.strip()
            msgs = question_messages(memory, mem_kind, q, day_names, cards, rooms, patrol_hours, look_on, hints, notes, None,
                                     fmt, patrol_times, question_moments, feedback_delay_min)
            if fmt == "conf":
                text, usage = ask(msgs, CONF_SCHEMA, 260, q.question_id, llm_authored=(notes,))
                loc, conf, why, status = parse_conf(text, allowed)
                ranking, look_room, carrier = ([loc] if loc else None), None, None
            else:
                text, usage = ask(msgs, ANSWER_SCHEMA_LOOK if look_on else ANSWER_SCHEMA, 120, q.question_id, llm_authored=(notes,))
                ranking, conf, look_room, carrier, status = parse_answer(text, allowed, set(rooms))
                why = ""
            rec: Dict[str, Any] = {**tag, "day_index": q.day_index, "question_id": q.question_id, "object_id": q.object_id,
                                   "object_class": q.object_class, "t_query": q.t_query, "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                                   "completion_tokens": int(usage.get("completion_tokens", 0)), "reasoning": why,
                                   "raw_confidence": conf, "patrol_label": label}
            fallback = ranking is None
            answer = ranking[0] if ranking else (memory.last_spot_real(q.object_id) or (sorted(allowed)[0] if fmt == "conf" else OUT_OF_HOUSE))
            top = conf if conf is not None and not fallback else 0.0
            rec.update({"answer_before_look": answer, "top_prob_before_look": round(top, 3), "status_before_look": status,
                        "asked_look": look_room})
            if look_on and look_room is not None and not fallback:
                t = q.t_query
                present = episode.residents_in_room(look_room, t)
                results = [SenseResult(receptacle_id=r, t=t, contents=episode.receptacle_contents(r, t),
                                       object_classes={o: episode.object_classes.get(o, "") for o in episode.receptacle_contents(r, t)},
                                       residents_present=present) for r in rooms[look_room]]
                memory.update_group(t, results, day_names, is_look=True)
                found = any(q.object_id in r.contents for r in results)
                who = ", ".join(names.get(p, p) for p in present) or "nobody"
                look_text = (f"You looked at the {look_room} just now ({clock(t, day_names)}); present: {who}. Contents: "
                             + "; ".join(f"{r.receptacle_id}: {', '.join(r.contents) if r.contents else 'empty'}" for r in results))
                msgs2 = question_messages(memory, mem_kind, q, day_names, cards, rooms, patrol_hours, look_on, hints, notes, look_text)
                text2, usage2 = ask(msgs2, ANSWER_SCHEMA, 120, q.question_id + " after look", llm_authored=(notes,))
                ranking2, conf2, _, carrier2, status2 = parse_answer(text2, allowed, set(rooms))
                rec.update({"look_room": look_room, "found_in_look": found, "status": status2,
                            "prompt_tokens": rec["prompt_tokens"] + int(usage2.get("prompt_tokens", 0))})
                if ranking2 is not None:
                    fallback, answer, top, carrier = False, ranking2[0], (conf2 if conf2 is not None else 0.0), carrier2
                else:
                    fallback, answer, top = True, (memory.last_spot_real(q.object_id) or OUT_OF_HOUSE), 0.0
            else:
                rec["status"] = status
            truth = episode.true_location(q.object_id, q.t_query)
            n_fallback += int(fallback)
            rec.update({"answer": answer, "top_prob": round(top, 3), "fallback": fallback,
                        "on_person_resident": carrier, "truth": truth,
                        "correct": answer == truth, "correct_before_look": rec["answer_before_look"] == truth,
                        "abstain_direct": answer == ABSTAIN})
            records.append(rec)
    calls.close()
    n_ok = sum(r["correct"] for r in records)
    n_abs = sum(r["abstain_direct"] for r in records)
    print(f"  {tag['household']} p{patrol_hours} {arm:36s} right {n_ok}/{len(records)}  direct abstain {n_abs}  "
          f"fallback {n_fallback}  calls {client.stats['calls']} cached {client.stats['cached']}", file=sys.stderr, flush=True)
    with open(run_dir / "run_log.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(run_dir / "noticing.jsonl", "w") as f:
        for r in noticing:
            f.write(json.dumps({**tag, "household": tag["household"], **r}, sort_keys=True) + "\n")
    (run_dir / "stats.json").write_text(json.dumps({"arm": arm, "n": len(records), "right": n_ok, "direct_abstain": n_abs,
                                                     "fallback": n_fallback, "dollars": 0.0, "model": client.model}, indent=1))
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--cache", type=pathlib.Path, default=None)
    ap.add_argument("--memory", nargs="+", default=list(MEMORIES), choices=MEMORIES)
    ap.add_argument("--told", nargs="+", default=["told", "not_told"], choices=["told", "not_told"])
    ap.add_argument("--look", nargs="+", default=["on", "off"], choices=["on", "off"])
    ap.add_argument("--max-days", type=int, default=None, help="stop after this day index (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replay-only", action="store_true")
    ap.add_argument("--format", default="rank", choices=("rank", "conf"))
    a = ap.parse_args(argv)
    if a.format == "conf":
        a.look = ["off"]
    client = LLMClient(a.cache or (a.out / "cache"), replay_only=a.replay_only)
    jobs = [(b, m, t == "told", l == "on") for b in a.bank for m in a.memory for t in a.told for l in a.look]
    # token estimate before the batch (local server: $0)
    est = len(jobs) * 224 * 1.4 * 1800
    print(f"{len(jobs)} arm runs; rough estimate {est / 1e6:.1f}M prompt tokens; hosted spend $0 (local vLLM {client.model})",
          file=sys.stderr, flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_arm, b, m, t, l, client, a.out, a.max_days, a.format) for b, m, t, l in jobs]

        def _report_now(job):
            # print the moment an arm dies, not when the ordered result loop below reaches it -- a dead arm in the
            # middle of a 30-arm pass otherwise looks like a hang for an hour (2026-09-22 00:27)
            def cb(f):
                e = f.exception()
                if e is not None:
                    print(f"ARM FAILED (live) {job[0].name} {job[1:]}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return cb
        for f, job in zip(futs, jobs):
            f.add_done_callback(_report_now(job))
        failed = 0
        for f, job in zip(futs, jobs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001 - one arm failing must not take the others down
                failed += 1
                print(f"ARM FAILED {job[0].name} {job[1:]}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"{failed} of {len(jobs)} arm runs failed", file=sys.stderr, flush=True)
    print(json.dumps(client.stats), file=sys.stderr)
    (a.out / "llm_stats.json").write_text(json.dumps({**client.stats, "dollars": 0.0, "model": client.model}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

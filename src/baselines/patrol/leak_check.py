"""Fail a run whose prompts contain hidden state.

Hidden state is anything only the generator and the harness know: the
truth rows' ``cause`` / ``causes`` / ``whim`` / ``reason`` fields, the
cause names from ``situation_sim/events.yaml`` and ``episodes.yaml``, the
internal-state flag names, and raw resident ids (prompts use names).

    python3 -m baselines.patrol.leak_check --calls results/.../llm/*/calls.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import sys
from typing import Iterable, List

import yaml

_SIM = pathlib.Path(__file__).resolve().parents[2] / "situation_sim"


class _StrictLoader(yaml.SafeLoader):
    """A YAML loader that rejects duplicate keys instead of silently
    keeping the last one."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for k_node, _ in node.value:
            k = self.construct_object(k_node, deep=deep)
            if k in seen:
                raise ValueError(f"duplicate YAML key {k!r} at {k_node.start_mark}")
            seen.add(k)
        return super().construct_mapping(node, deep)


def forbidden_terms() -> List[str]:
    events = yaml.load((_SIM / "events.yaml").read_text(), Loader=_StrictLoader)["events"]
    episodes = yaml.load((_SIM / "episodes.yaml").read_text(), Loader=_StrictLoader)["episodes"]
    terms = ["cause", "causes", "whim", "whim_p", "reason", "hidden_state",
             "low_energy", "running_late", "distracted", "episode_id"]
    terms += sorted(events) + sorted(episodes)
    return sorted(set(terms))


def leak_pattern() -> re.Pattern:
    terms = forbidden_terms()
    return re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(t) for t in terms) + r")(?![A-Za-z0-9_])"
                      r"|(?<![A-Za-z0-9_])resident_\d+(?![A-Za-z0-9_])")


PATTERN = leak_pattern()
PROSE_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(
    re.escape(t) for t in forbidden_terms() if "_" in t or t == "whim")
    + r")(?![A-Za-z0-9_])|(?<![A-Za-z0-9_])resident_\d+(?![A-Za-z0-9_])")
"""For text the LLM wrote itself (its own routine notes, quoted back into
a later prompt): plain words such as "rain" or "deadline" are ordinary
prose there, so only the structural ids count -- snake_case event and
field names, resident ids, and "whim". The truth-row field names "cause"
/ "causes" / "reason" are also everyday English and were tripping this
on model-written sentences like "the likely cause of the error"
(2026-09-22, reflection arm hh_s5, day 1): the model never sees those
fields, so its own use of the words cannot be a leak; they stay
forbidden in harness-authored text (PATTERN above)."""


def find_leaks(text: str, prose: bool = False) -> List[str]:
    pat = PROSE_PATTERN if prose else PATTERN
    return sorted({m.group(0) for m in pat.finditer(text)})


def check_prompt(messages: Iterable[dict], where: str = "", llm_authored: Iterable[str] = ()) -> None:
    """Raise if any message of a prompt contains a forbidden term. Text in
    ``llm_authored`` (the model's own earlier output, quoted in the prompt)
    is checked with the prose pattern; everything else strictly."""
    for m in messages:
        text = m.get("content", "")
        for span in llm_authored:
            if span:
                text = text.replace(span, "")
        hits = find_leaks(text)
        if hits:
            raise RuntimeError(f"LEAK in prompt {where}: {hits}")
    for span in llm_authored:
        hits = find_leaks(span or "", prose=True)
        if hits:
            raise RuntimeError(f"LEAK in LLM-authored notes {where}: {hits}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", nargs="+", required=True, help="calls.jsonl files (one prompt per line)")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    files = sorted(p for pat in a.calls for p in glob.glob(pat))
    n = 0
    bad = []
    for f in files:
        with open(f) as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                rec = json.loads(line)
                n += 1
                spans = rec.get("llm_authored", [])
                for m in rec["messages"]:
                    text = m.get("content", "")
                    for span in spans:
                        text = text.replace(span, "")
                    hits = find_leaks(text) + find_leaks(" ".join(spans), prose=True)
                    if hits:
                        bad.append({"file": f, "line": i + 1, "hits": hits})
    text = [f"# Leak check", "", f"{n} prompts in {len(files)} files scanned for {len(forbidden_terms())} forbidden terms.",
            f"Leaks found: {len(bad)}", ""]
    for b in bad[:50]:
        text.append(f"- {b['file']}:{b['line']}: {b['hits']}")
    if a.out:
        a.out.write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

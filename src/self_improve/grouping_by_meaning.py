"""Which two notes might be saying the same thing, judged by meaning rather than shared words.

This is the grouping half of ACE's grow-and-refine step. In their code the grouping is done
with sentence embeddings (all-mpnet-base-v2) and cosine similarity at 0.90, and the MERGE is
then written by a language model. Ours had it exactly the other way round - shared words to
group, arithmetic to merge - so this module and the model-written merge beside it are what
make that step faithful in shape.

WHY THERE IS NO 0.90 HERE. Their threshold belongs to their embedding model. Measured on this
machine with the only model available offline (Llama-3.2-1B, mean-pooled, on the processor),
unrelated sentences already score 0.83 to 0.88 and true paraphrases score 0.92 to 0.96 - a
much higher floor and a much thinner margin than a sentence-embedding model gives. Borrowing
0.90 onto that scale would be the same mistake as every other borrowed number in this project.

So the embedding does what it is actually good for: RANKING. It proposes the few most similar
pairs, and the model is asked whether each pair really says one thing. That keeps their
division of labour - cheap grouping, model judgement - without pretending a threshold from a
different scale transfers. The cosine values are recorded so a threshold can be calibrated
later from real pairs rather than assumed.
"""
from __future__ import annotations

import itertools
import os
from typing import Any, List, Optional, Sequence, Tuple

WHICH_MODEL = "meta-llama/Llama-3.2-1B"
HOW_MANY_PAIRS_TO_PROPOSE = 4

_loaded: List[Any] = []


def _model():
    """Loaded once, lazily, on the processor. The GPU is busy serving the run."""
    if _loaded:
        return _loaded[0], _loaded[1]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(WHICH_MODEL)
    tok.pad_token = tok.eos_token
    mod = AutoModel.from_pretrained(WHICH_MODEL, dtype=torch.float32)
    mod.eval()
    _loaded.extend([tok, mod])
    return tok, mod


def how_alike(statements: Sequence[str]) -> Any:
    """A unit-length vector per statement, mean-pooled over its tokens."""
    import torch
    tok, mod = _model()
    with torch.no_grad():
        batch = tok(list(statements), return_tensors="pt", padding=True, truncation=True,
                    max_length=128)
        hidden = mod(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(pooled, dim=-1)


def pairs_worth_asking_about(claims: Sequence[Any],
                            how_many: int = HOW_MANY_PAIRS_TO_PROPOSE
                            ) -> List[Tuple[Any, Any, float]]:
    """The most similar pairs of notes, most similar first, with their cosine.

    Only pairs that hold under the same condition and share an object are proposed, for the
    reason recorded in write_the_notes_told_if_right: the pair this study exists to keep
    apart - one object under two routines - is exactly the pair a similarity score likes
    most, and merging it would destroy the thing being measured. So meaning decides the
    ORDER and those two rules decide what is eligible at all.
    """
    live = [c for c in claims if getattr(c, "folded_into", None) is None]
    if len(live) < 2:
        return []
    vectors = how_alike([c.statement for c in live])
    out: List[Tuple[Any, Any, float]] = []
    for i, j in itertools.combinations(range(len(live)), 2):
        one, two = live[i], live[j]
        if one.holds_under.strip().lower() != two.holds_under.strip().lower():
            continue
        out.append((one, two, float(vectors[i] @ vectors[j])))
    out.sort(key=lambda row: -row[2])
    return out[:how_many]

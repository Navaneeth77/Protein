"""P2.1 + P2.2 — ESM-2 sequence plausibility signals.

Agent-visible. The only inputs are a candidate sequence string; nothing in this
module can reach the evaluator-only data tree.

Provides:
    masked_marginal_matrix(seq) -> (L, 20) probabilities over the standard AAs
    residue_surprisal(seq)      -> (L,) negative log-likelihood per position
    pseudo_log_likelihood(seq)  -> float, sum of per-position log-likelihoods
    rank_suspicious_positions(seq, top_k) -> list[int]

Masked-marginal scoring needs one forward pass per position, so results are
memoised in-process and on disk keyed by a sequence hash. That keeps the demo
path off the model (constraint C4).
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import numpy as np

from ..constants import AA_ALPHABET, ESM2_SCORE_MODEL
from ..paths import CACHE

SCORE_CACHE = CACHE / "esm_score"
BATCH_SIZE = 16
OFFLINE_ENV = "REFOLD_OFFLINE"


class ScoringUnavailable(RuntimeError):
    """Raised when a score is not cached and live model use is disabled."""


_MODEL = None
_TOKENIZER = None
_AA_TOKEN_IDS: np.ndarray | None = None

# Instrumentation, mirrored on the fold cache so the replay-mode assertion in
# P5.1 can prove nothing hit a model.
STATS = {"cache_hits": 0, "cache_misses": 0, "forward_batches": 0}


def reset_stats() -> None:
    for k in STATS:
        STATS[k] = 0


def offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "") not in ("", "0", "false", "False")


def _load():
    """Lazily load the masked-LM scorer. Never called on a cache hit."""
    global _MODEL, _TOKENIZER, _AA_TOKEN_IDS
    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _AA_TOKEN_IDS

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(ESM2_SCORE_MODEL)
    _MODEL = AutoModelForMaskedLM.from_pretrained(ESM2_SCORE_MODEL)
    _MODEL.eval()
    torch.set_grad_enabled(False)
    _AA_TOKEN_IDS = np.array(
        [_TOKENIZER.convert_tokens_to_ids(aa) for aa in AA_ALPHABET], dtype=int
    )
    return _MODEL, _TOKENIZER, _AA_TOKEN_IDS


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def _cache_path(sequence: str):
    return SCORE_CACHE / f"{sequence_hash(sequence)}.npy"


def masked_marginal_matrix(sequence: str) -> np.ndarray:
    """(L, 20) probability of each standard amino acid at each position.

    Column order follows `AA_ALPHABET`. Each row comes from a forward pass with
    that single position replaced by <mask>, i.e. the masked-marginal estimator
    from Meier et al. rather than the cheaper wild-type-marginal shortcut.
    """
    cached = _load_cached_matrix(sequence)
    if cached is not None:
        return cached

    if offline():
        raise ScoringUnavailable(
            f"masked-marginal matrix for {sequence_hash(sequence)} is not cached "
            f"and {OFFLINE_ENV} is set"
        )

    import torch

    model, tokenizer, aa_ids = _load()
    length = len(sequence)
    encoded = tokenizer(sequence, return_tensors="pt")
    base_ids = encoded["input_ids"][0]
    attention = encoded["attention_mask"][0]
    # ESM prepends <cls>, so residue i lives at token i + 1.
    offset = 1
    probs = np.zeros((length, len(AA_ALPHABET)), dtype=np.float64)

    for start in range(0, length, BATCH_SIZE):
        positions = list(range(start, min(start + BATCH_SIZE, length)))
        batch = base_ids.repeat(len(positions), 1)
        for row, pos in enumerate(positions):
            batch[row, pos + offset] = tokenizer.mask_token_id
        out = model(
            input_ids=batch,
            attention_mask=attention.repeat(len(positions), 1),
        ).logits
        STATS["forward_batches"] += 1
        for row, pos in enumerate(positions):
            logits = out[row, pos + offset]
            aa_logits = logits[torch.as_tensor(aa_ids)]
            probs[pos] = torch.softmax(aa_logits.double(), dim=-1).numpy()

    STATS["cache_misses"] += 1
    _store_cached_matrix(sequence, probs)
    return probs


def _load_cached_matrix(sequence: str) -> np.ndarray | None:
    path = _cache_path(sequence)
    if path.exists():
        STATS["cache_hits"] += 1
        return np.load(path)
    return None


def _store_cached_matrix(sequence: str, probs: np.ndarray) -> None:
    SCORE_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(_cache_path(sequence), probs)


@lru_cache(maxsize=512)
def _log_probs_of_sequence(sequence: str) -> tuple:
    probs = masked_marginal_matrix(sequence)
    idx = np.array([AA_ALPHABET.index(a) for a in sequence])
    picked = probs[np.arange(len(sequence)), idx]
    return tuple(np.log(np.clip(picked, 1e-12, None)).tolist())


def residue_surprisal(sequence: str) -> np.ndarray:
    """Per-position negative log-likelihood of the observed residue."""
    return -np.array(_log_probs_of_sequence(sequence), dtype=float)


def pseudo_log_likelihood(sequence: str) -> float:
    """Sum of per-position masked log-likelihoods.

    Unbounded and negative. Anything combining this with a 0-1 quantity must
    rescale first (see the evaluator's `esm_score` note).
    """
    return float(np.sum(_log_probs_of_sequence(sequence)))


def mean_surprisal(sequence: str) -> float:
    return float(residue_surprisal(sequence).mean())


def rank_suspicious_positions(sequence: str, top_k: int = 3) -> list[int]:
    """0-based positions with the highest surprisal, most suspicious first.

    Ties break toward the lower index so the whole pipeline stays deterministic.
    """
    surprisal = residue_surprisal(sequence)
    order = np.lexsort((np.arange(len(sequence)), -surprisal))
    return [int(i) for i in order[:top_k]]


def substitution_ranking(
    sequence: str, position: int, top_n: int
) -> list[tuple[str, float]]:
    """Top-n amino acids at `position` by masked-marginal probability.

    The residue currently at `position` is excluded — a "substitution" that
    changes nothing is not a candidate.
    """
    probs = masked_marginal_matrix(sequence)[position]
    current = sequence[position]
    ranked = sorted(
        ((aa, float(p)) for aa, p in zip(AA_ALPHABET, probs) if aa != current),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return ranked[:top_n]

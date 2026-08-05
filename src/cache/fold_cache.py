"""P2.5 — cached ESMFold structure prediction plus structural feature extraction.

Every structure the agent ever sees comes through here. The cache is keyed by a
hash of the sequence alone, so a demo replay resolves entirely from disk with the
model never loaded (constraint C4). Set REFOLD_OFFLINE=1 to make an unexpected
cache miss a loud error instead of a silent model download.

pLDDT note: high pLDDT is *model confidence*, not correctness. This module makes
no claim about whether a fold is right — only src/evaluator.py, which holds the
reference structure, decides that.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..constants import CONTACT_CUTOFF_ANGSTROM, ESMFOLD_MODEL, LONG_RANGE_SEPARATION
from ..geometry import (
    clash_count,
    contact_degrees,
    contact_set,
    contiguous_regions,
    radius_of_gyration,
    secondary_structure,
)
from ..paths import CACHE
from ..pdb_io import first_chain

OFFLINE_ENV = "REFOLD_OFFLINE"
BACKEND_ENV = "REFOLD_FOLD_BACKEND"
LOCAL_PATH_ENV = "REFOLD_ESMFOLD_PATH"

ESMFOLD_BACKEND = "esmfold"
SYNTHETIC_BACKEND = "synthetic"

STATS = {"cache_hits": 0, "cache_misses": 0, "model_calls": 0}

_MODEL = None
_TOKENIZER = None


class FoldUnavailable(RuntimeError):
    """A structure was requested that is neither cached nor allowed to be run."""


@dataclass
class FoldResult:
    sequence: str
    pdb_text: str
    plddt: np.ndarray                 # per-residue, rescaled to 0-1
    plddt_raw_range: tuple            # observed (min, max) before rescaling
    mean_plddt: float
    contacts: set = field(default_factory=set)
    contact_degree: np.ndarray = None
    long_range_contact_degree: np.ndarray = None
    ss_labels: list = field(default_factory=list)
    helices: list = field(default_factory=list)
    strands: list = field(default_factory=list)
    radius_of_gyration: float = 0.0
    clashes: int = 0
    ca_coords: np.ndarray = None
    cb_coords: np.ndarray = None
    from_cache: bool = False
    # True when the coordinates came from src/cache/synthetic_backend.py, i.e.
    # they are a harness fixture and not a structure prediction. Anything that
    # displays or reports a score must surface this.
    synthetic: bool = False


def reset_stats() -> None:
    for k in STATS:
        STATS[k] = 0


def offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "") not in ("", "0", "false", "False")


def backend() -> str:
    """Which structure source is in use. Read at call time, never cached."""
    value = os.environ.get(BACKEND_ENV, ESMFOLD_BACKEND).strip().lower()
    if value not in (ESMFOLD_BACKEND, SYNTHETIC_BACKEND):
        raise ValueError(
            f"{BACKEND_ENV}={value!r} is not one of "
            f"{ESMFOLD_BACKEND!r}, {SYNTHETIC_BACKEND!r}"
        )
    return value


def using_synthetic_backend() -> bool:
    return backend() == SYNTHETIC_BACKEND


def checkpoint_path() -> str:
    """Local checkpoint directory if one is configured, else the hub id."""
    return os.environ.get(LOCAL_PATH_ENV, "").strip() or ESMFOLD_MODEL


def sequence_hash(sequence: str) -> str:
    """Stable cache key. Truncated sha256 — collision risk is nil at this scale."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def cache_dir():
    """Where structures for the ACTIVE backend live.

    Synthetic structures get their own subdirectory. They are a harness fixture,
    and if they shared a directory with real predictions a run configured for
    ESMFold would silently be served a fixture — which is exactly the mistake
    this split prevents.
    """
    return CACHE / "synthetic" if using_synthetic_backend() else CACHE


def cache_path(sequence: str):
    return cache_dir() / f"{sequence_hash(sequence)}.pdb"


def is_cached(sequence: str) -> bool:
    return cache_path(sequence).exists()


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    source = checkpoint_path()

    # Memory, and why this is not the obvious `from_pretrained(source)`:
    # the fp32 checkpoint is 8.4 GB and this machine has ~4.7 GB free, so a
    # plain load is OOM-killed during weight materialisation. Loading straight
    # into bfloat16 halves it to ~4.2 GB, and the folding trunk — a small
    # fraction of the parameters, and the part that actually builds coordinates,
    # where reduced precision would show up as geometry error — is then upcast
    # back to fp32. bfloat16 rather than float16 because torch 2.x has far
    # better CPU kernel coverage for it.
    # Override with REFOLD_FOLD_DTYPE=float32 on a machine with the headroom.
    dtype_name = os.environ.get("REFOLD_FOLD_DTYPE", "bfloat16").strip().lower()
    dtype = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"REFOLD_FOLD_DTYPE={dtype_name!r} is not a supported dtype")

    print(f"[fold] loading {source} as {dtype_name} (first call only)")
    _TOKENIZER = AutoTokenizer.from_pretrained(source)
    _MODEL = EsmForProteinFolding.from_pretrained(
        source, low_cpu_mem_usage=True, dtype=dtype
    )

    if dtype is not torch.float32:
        # Upcast the structure side back to fp32. The boundary is safe because
        # EsmForProteinFolding.forward casts the language-model output with
        # `esm_s.to(self.esm_s_combine.dtype)`, so making esm_s_combine fp32 is
        # what promotes the activations. Modules cast in place; bare Parameters
        # have to be reassigned through `.data`.
        for attribute in ("trunk", "esm_s_mlp", "embedding", "lm_head", "distogram_head",
                          "ptm_head", "lddt_head"):
            target = getattr(_MODEL, attribute, None)
            if isinstance(target, torch.nn.Module):
                target.to(torch.float32)
        for attribute in ("esm_s_combine", "af2_to_esm"):
            param = getattr(_MODEL, attribute, None)
            if isinstance(param, torch.nn.Parameter):
                param.data = param.data.to(torch.float32)
            elif isinstance(param, torch.Tensor) and param.is_floating_point():
                setattr(_MODEL, attribute, param.to(torch.float32))
        print("[fold] folding trunk and heads upcast to fp32")

    _MODEL.eval()

    # Chunking trades a little speed for a much lower memory peak in attention.
    if hasattr(_MODEL, "trunk"):
        _MODEL.trunk.set_chunk_size(64)

    torch.set_grad_enabled(False)
    return _MODEL, _TOKENIZER


def _predict_pdb(sequence: str) -> str:
    """Produce coordinates for `sequence` using the configured backend."""
    if using_synthetic_backend():
        from . import synthetic_backend

        STATS["model_calls"] += 1
        return synthetic_backend.predict(sequence)

    model, tokenizer = _load_model()
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    outputs = model(**inputs)
    STATS["model_calls"] += 1
    return model.output_to_pdb(outputs)[0]


def fold(sequence: str) -> FoldResult:
    """Predicted structure + features for `sequence`, cached on disk."""
    path = cache_path(sequence)
    if path.exists():
        STATS["cache_hits"] += 1
        return structure_features(
            sequence, path.read_text(encoding="utf-8"), from_cache=True
        )

    if offline():
        raise FoldUnavailable(
            f"no cached structure for {sequence_hash(sequence)} and {OFFLINE_ENV} "
            f"is set — run scripts/research/precompute.py first"
        )

    STATS["cache_misses"] += 1
    pdb_text = _predict_pdb(sequence)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pdb_text, encoding="utf-8")
    _append_index(sequence)
    return structure_features(sequence, pdb_text, from_cache=False)


def _append_index(sequence: str) -> None:
    """Human-readable map from hash back to sequence, for debugging the cache."""
    index = CACHE / "index.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"hash": sequence_hash(sequence), "sequence": sequence}) + "\n")


def structure_features(
    sequence: str, pdb_text: str, *, from_cache: bool = False
) -> FoldResult:
    """Extract every structural feature the agent is allowed to see."""
    chain = first_chain(pdb_text, is_text=True)
    if len(chain) != len(sequence):
        raise ValueError(
            f"predicted structure has {len(chain)} residues, sequence has {len(sequence)}"
        )

    raw_plddt = chain.ca_bfactors()
    raw_range = (float(raw_plddt.min()), float(raw_plddt.max()))
    # ESMFold writes pLDDT into the B-factor column. The scale is checked here
    # rather than assumed: values above 1.5 mean the 0-100 convention.
    plddt = raw_plddt / 100.0 if raw_range[1] > 1.5 else raw_plddt

    ca = chain.ca_coords()
    cb = chain.cb_coords()
    contacts = contact_set(cb, cutoff=CONTACT_CUTOFF_ANGSTROM)
    total_deg, long_deg = contact_degrees(contacts, len(chain), LONG_RANGE_SEPARATION)
    labels = secondary_structure(ca, contacts)
    heavy_coords, heavy_owner = chain.heavy_atoms()

    from . import synthetic_backend

    return FoldResult(
        sequence=sequence,
        pdb_text=pdb_text,
        synthetic=synthetic_backend.is_synthetic(pdb_text),
        plddt=plddt,
        plddt_raw_range=raw_range,
        mean_plddt=float(plddt.mean()),
        contacts=contacts,
        contact_degree=total_deg,
        long_range_contact_degree=long_deg,
        ss_labels=labels,
        helices=contiguous_regions(labels, "helix", min_length=4),
        strands=contiguous_regions(labels, "strand", min_length=2),
        radius_of_gyration=radius_of_gyration(heavy_coords),
        clashes=clash_count(heavy_coords, heavy_owner),
        ca_coords=ca,
        cb_coords=cb,
        from_cache=from_cache,
    )

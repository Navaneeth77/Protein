"""P3.5 — the Gemma client and bounded patch validation.

The outer loop's ONLY output channel is a policy patch expressed in the fixed DSL
of P3.1 (constraint C2). Gemma never emits code, never names a file, and never
gets to invent a computation: a patch may re-weight a `state.json` field that the
grounder already computes (a *representation* patch) or change one of four
proposal fields (a *mechanism* patch). Anything else is rejected and logged.

Transport is configurable via environment:
    REFOLD_GEMMA_MODE   mock (default) | openai | ollama
    REFOLD_GEMMA_URL    endpoint for openai/ollama modes
    REFOLD_GEMMA_MODEL  model id, e.g. gemma-3-27b-it / gemma3:12b
    REFOLD_GEMMA_KEY    bearer token for openai-compatible endpoints

`mock` mode is a scripted proposer, not a language model. It exists so the demo
can run with no endpoint reachable; anything it produces is labelled
`source: "mock"` in the logs so a scripted patch can never be mistaken for a
model-authored one.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..paths import LOGS
from . import grounder, policy as policy_mod

CALL_LOG = LOGS / "gemma_calls.jsonl"

MECHANISM_FIELDS = (
    "positions",
    "substitutions_per_position",
    "preserve_residue_class",
    "max_total_edits",
)

# Shape handed to Ollama's structured-output mode so the reply cannot come back
# missing `kind` or wrapped in prose. Intentionally loose about the *values* —
# the real gate is policy validation in apply_patch, not this.
PATCH_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["representation", "mechanism"]},
        "rationale": {"type": "string"},
        "position_score": {
            "type": "object",
            "properties": {
                "esm_surprisal": {"type": "number"},
                "low_plddt": {"type": "number"},
                "contact_violation": {"type": "number"},
                "long_range_contact_violation": {"type": "number"},
            },
            "additionalProperties": False,
        },
        # max_total_edits is deliberately absent. Constrained decoding enforces
        # the *shape* but not numeric bounds, and the model kept answering
        # max_total_edits = 5000000000000000, which the policy schema then threw
        # out — burning a whole attempt. It also has no effect on a single-round
        # repair, so removing it from the action space costs nothing and removes
        # the failure mode without coercing anything the model said.
        "proposal": {
            "type": "object",
            "properties": {
                "positions": {"type": "integer", "minimum": 1, "maximum": 10},
                "substitutions_per_position": {
                    "type": "integer", "minimum": 1, "maximum": 19,
                },
                "preserve_residue_class": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    "required": ["kind", "rationale"],
}

SYSTEM_PROMPT = """You are the outer loop of a protein-repair search harness.

You may reply with EXACTLY ONE JSON object and nothing else. No prose, no code,
no markdown outside the JSON. Schema:

{
  "kind": "representation" | "mechanism",
  "rationale": "<one or two sentences>",
  "position_score": { "<feature>": <weight 0..1>, ... },   // representation only
  "proposal": { "<field>": <value>, ... }                  // mechanism only
}

Rules you must obey or your patch is discarded:
- "position_score" keys must be drawn from ALLOWED_FEATURES given below. You may
  not introduce any other key, and you may not describe a new computation.
- "position_score" weights must sum to exactly 1.0.
- "proposal" keys must be drawn from ALLOWED_PROPOSAL_FIELDS given below.
- Change one thing. A representation patch adjusts weights only; a mechanism
  patch adjusts proposal fields only.

Your patch must CHANGE WHAT THE SEARCH DOES. A patch that parses but leaves the
same positions selected and the same substitutions proposed is worthless. Before
answering, check your patch against CURRENTLY_SELECTED_POSITIONS and the residue
table: either shift enough weight onto a feature that ranks different residues
highly, or change a proposal field so the search widens or unblocks.
"""


class PatchRejected(ValueError):
    """A proposed patch failed validation and was discarded."""


@dataclass
class PatchOutcome:
    patch: dict | None
    policy: dict | None
    raw: str
    source: str
    error: str | None = None

    @property
    def accepted(self) -> bool:
        return self.patch is not None and self.policy is not None


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #

def build_prompt(
    current_policy: dict,
    candidate_outcomes: list[dict],
    counterexample: dict | None,
    state_schema: dict | None = None,
    focus: dict | None = None,
) -> str:
    """Assemble the prompt.

    `focus` is optional extra grounding for the MVP path: the positions the
    current policy selected, a small table of per-residue feature values, and
    (on a retry) the fact that a previous patch changed nothing. Giving the model
    the actual numbers is what lets it propose a patch with real consequences
    rather than a plausible-sounding no-op.
    """
    lines = [
        "ALLOWED_FEATURES (state.json residue fields you may weight):",
        json.dumps(sorted(grounder.SCORABLE_FEATURES)),
        "",
        # Ranges spelled out because the model otherwise emits things like
        # max_total_edits = 3000000000000000, which the schema then rejects.
        "ALLOWED_PROPOSAL_FIELDS, with the ONLY legal values for each:",
        "  positions:                   integer 1..10   (currently 3)",
        "  substitutions_per_position:  integer 1..19   (currently 4)",
        "  preserve_residue_class:      true or false   (currently true)",
        "Any value outside these ranges causes your patch to be discarded. Use "
        "small integers; do not write large numbers.",
        "",
        "CURRENT POLICY:",
        policy_mod.dump_policy(current_policy),
        "",
        "CANDIDATE OUTCOMES THIS GENERATION (predicted vs observed):",
    ]
    for run in candidate_outcomes:
        lines.append(
            json.dumps(
                {
                    "variant": run.get("variant"),
                    "mutations": run.get("mutations"),
                    "predicted_delta": round(float(run.get("predicted_delta", 0.0)), 6),
                    "observed_hidden_delta": round(float(run.get("hidden_delta", 0.0)), 6),
                    "mean_plddt": round(float(run.get("mean_plddt", 0.0)), 6),
                    "esm_score": round(float(run.get("esm_score", 0.0)), 6),
                }
            )
        )

    if counterexample:
        # Summary only, no state_before dump. The full grounded state ran to
        # thousands of tokens and pushed the prompt past Ollama's default 4096
        # context; the residue table below carries the same signal far cheaper.
        lines += [
            "",
            "COUNTEREXAMPLE (public signals improved, hidden verifier did not):",
            json.dumps(
                {
                    "iteration": counterexample.get("iteration"),
                    "predicted_delta": counterexample.get("predicted_delta"),
                    "hidden_delta": counterexample.get("hidden_delta"),
                    "variant": counterexample.get("variant"),
                    "mutations": counterexample.get("mutations"),
                    "summary": (counterexample.get("state_before") or {}).get("summary"),
                }
            ),
        ]

    if focus:
        if focus.get("selected_positions") is not None:
            lines += [
                "",
                "CURRENTLY_SELECTED_POSITIONS (1-based) — your patch should not "
                "leave this list unchanged:",
                json.dumps([p + 1 for p in focus["selected_positions"]]),
            ]
        if focus.get("residues"):
            lines += [
                "",
                "RESIDUE FEATURE TABLE (the values your weights multiply; "
                "1-based position):",
                "position  aa  esm_surprisal  low_plddt  contact_violation  "
                "long_range_contact_violation",
            ]
            for residue in focus["residues"]:
                lines.append(
                    f"{residue['position'] + 1:>8}  {residue['aa']:>2}  "
                    f"{residue['esm_surprisal']:>13.3f}  {residue['low_plddt']:>9.3f}  "
                    f"{residue['contact_violation']:>17.3f}  "
                    f"{residue['long_range_contact_violation']:>28.3f}"
                )
        if focus.get("previous_attempt_changed_nothing"):
            lines += [
                "",
                "IMPORTANT: your previous patch was valid but selected exactly the "
                "same positions and proposed exactly the same substitutions, so it "
                "did nothing. The two changes that reliably alter the search are: "
                "set `proposal.positions` to 5 (examines two more sites), or set "
                "`preserve_residue_class` to false (unblocks substitutions across "
                "residue classes). Pick one of those, or move most of the "
                "position_score weight onto a feature that ranks different "
                "residues highly. Keep every value inside its allowed range.",
            ]

    lines += [
        "",
        "Propose ONE patch that would make the next generation's hidden score higher.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# transports
# --------------------------------------------------------------------------- #

def mode() -> str:
    return os.environ.get("REFOLD_GEMMA_MODE", "mock").lower()


def _call_openai(prompt: str) -> str:
    import urllib.request

    url = os.environ.get("REFOLD_GEMMA_URL", "http://localhost:8000/v1/chat/completions")
    model = os.environ.get("REFOLD_GEMMA_MODEL", "gemma-3-27b-it")
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    key = os.environ.get("REFOLD_GEMMA_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


def _call_ollama(prompt: str) -> str:
    import urllib.request

    url = os.environ.get("REFOLD_GEMMA_URL", "http://localhost:11434/api/generate")
    model = os.environ.get("REFOLD_GEMMA_MODEL", "gemma4:12b")
    # keep_alive: how long Ollama holds the weights after answering. NOT 0 by
    # default — with keep_alive=0 this setup reliably wedged with the model stuck
    # in "Stopping..." while the HTTP response never arrived, so the unload races
    # the reply. 60s is long enough to avoid that and short enough that the ~8.9 GB
    # is released before anything else needs it (ESMFold peaks near 8.4 GB on a
    # 14 GB box, and the demo path serves structures from cache anyway).
    keep_alive = os.environ.get("REFOLD_GEMMA_KEEP_ALIVE", "60s")
    # Performance, all three of these learned the hard way on this box:
    #  * num_gpu: a 12B model does not fit a 4 GB card. Ollama's automatic split
    #    OOM'd ("CUDA error: out of memory") and wedged llama-server, but running
    #    fully on CPU took over ten minutes for one call. Offloading a fixed,
    #    conservative number of layers is the middle ground; on CUDA OOM we
    #    retry once on CPU rather than failing the demo.
    #  * num_predict: the answer is one small JSON object. Without a cap the
    #    model can ramble for thousands of tokens at CPU speed.
    #  * num_ctx: 4096 is Ollama's default and enough for this prompt now that
    #    the grounded-state dump is summarised out.
    num_gpu = int(os.environ.get("REFOLD_GEMMA_NUM_GPU", "14"))
    timeout = float(os.environ.get("REFOLD_GEMMA_TIMEOUT", "600"))

    def call(gpu_layers: int) -> str:
        payload = {
            "model": model,
            "prompt": SYSTEM_PROMPT + "\n\n" + prompt,
            "stream": False,
            # Constrained decoding against the patch shape. Plain "json" was not
            # enough: the model produced valid JSON but kept omitting "kind", so
            # the schema pins the envelope too. Without any constraint a 12B model
            # opens with prose and num_predict truncates it before the JSON, which
            # showed up as "no JSON object found in reply".
            "format": PATCH_FORMAT_SCHEMA,
            "keep_alive": int(keep_alive) if keep_alive.lstrip("-").isdigit() else keep_alive,
            "options": {
                "temperature": 0.2,
                "num_gpu": gpu_layers,
                "num_ctx": 4096,
                "num_predict": 500,
            },
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["response"]

    try:
        return call(num_gpu)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        if num_gpu == 0 or "memory" not in detail.lower():
            raise RuntimeError(f"ollama HTTP {exc.code}: {detail}") from exc
        print(f"[gemma] GPU offload failed ({detail}); retrying on CPU only")
        return call(0)


def _call_mock(prompt: str, current_policy: dict) -> str:
    """Scripted proposer. NOT a language model — see module docstring.

    Heuristic: if long-range contact information is available but unweighted,
    activate it; otherwise widen the proposal a little.
    """
    weights = current_policy["position_score"]
    if "long_range_contact_violation" not in weights:
        patch = {
            "kind": "representation",
            "rationale": (
                "Public signals improved while the hidden verifier did not, and the "
                "chosen edits sat at positions with intact local packing. Activate "
                "long_range_contact_violation so long-range contact loss is scored."
            ),
            "position_score": {
                "esm_surprisal": 0.45,
                "low_plddt": 0.15,
                "contact_violation": 0.15,
                "long_range_contact_violation": 0.25,
            },
        }
    else:
        proposal = current_policy["proposal"]
        patch = {
            "kind": "mechanism",
            "rationale": "Long-range term is already active; widen the site search instead.",
            "proposal": {"positions": min(int(proposal["positions"]) + 1, 10)},
        }
    return json.dumps(patch)


# --------------------------------------------------------------------------- #
# parsing, validation, application
# --------------------------------------------------------------------------- #

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_patch(raw: str) -> dict:
    """Extract the single JSON object from a model reply.

    Tolerant about markdown fences and surrounding prose because those are
    formatting noise, strict about everything that affects behaviour.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text.strip())
    match = _JSON_BLOCK.search(text)
    if not match:
        raise PatchRejected("no JSON object found in reply")
    try:
        patch = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise PatchRejected(f"reply is not valid JSON: {exc}") from exc
    if not isinstance(patch, dict):
        raise PatchRejected("patch must be a JSON object")
    return patch


def infer_kind(patch: dict) -> str:
    """What kind of patch is this, whether or not the model said so.

    Local models routinely omit `kind` or set it inconsistently with the keys
    they actually sent. The keys are the ground truth, so they decide. Note this
    relaxes only a presentation convention: every field still has to survive the
    P3.1 schema below, which is what constraint C2 actually rests on.
    """
    has_weights = isinstance(patch.get("position_score"), dict) and patch["position_score"]
    has_proposal = isinstance(patch.get("proposal"), dict) and patch["proposal"]
    if has_weights and has_proposal:
        return "combined"
    if has_weights:
        return "representation"
    if has_proposal:
        return "mechanism"
    raise PatchRejected(
        "patch carries neither a position_score nor a proposal, so it changes nothing"
    )


def describe_patch(patch: dict | None) -> str | None:
    """The kind a patch actually is, for display. None if there is no patch."""
    if not patch:
        return None
    try:
        return infer_kind(patch)
    except PatchRejected:
        return patch.get("kind")


def apply_patch(current_policy: dict, patch: dict) -> dict:
    """Return the patched policy, or raise. Never coerces an invalid payload."""
    declared = patch.get("kind")
    actual = infer_kind(patch)

    # A declared kind is honoured when it is consistent with the keys present.
    # Declaring "mechanism" while sending only weights is a contradiction, not a
    # patch, so it is rejected rather than silently reinterpreted.
    if declared in ("representation", "mechanism") and actual != "combined":
        if declared != actual:
            raise PatchRejected(
                f"patch declares kind {declared!r} but carries a {actual} change"
            )

    new_policy = policy_mod.clone(current_policy)

    if actual in ("representation", "combined"):
        weights = patch["position_score"]
        unknown = sorted(set(weights) - set(grounder.SCORABLE_FEATURES))
        if unknown:
            raise PatchRejected(
                f"position_score names feature(s) absent from state.json: {unknown}"
            )
        new_policy["position_score"] = {k: float(v) for k, v in weights.items()}

    if actual in ("mechanism", "combined"):
        proposal = patch["proposal"]
        unknown = sorted(set(proposal) - set(MECHANISM_FIELDS))
        if unknown:
            raise PatchRejected(f"proposal names field(s) not in the DSL: {unknown}")
        new_policy["proposal"].update(proposal)

    # Final gate: the P3.1 schema, including the sum-to-1.0 and known-feature
    # rules. This is what the interpreter is protected by.
    return policy_mod.validate_policy(new_policy)


def _log_call(record: dict, path: Path | None = None) -> None:
    path = Path(path) if path else CALL_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, allow_nan=False) + "\n")


def propose_patch(
    current_policy: dict,
    candidate_outcomes: list[dict],
    counterexample: dict | None = None,
    transport=None,
    log_path: Path | None = None,
    focus: dict | None = None,
) -> PatchOutcome:
    """Ask for one patch, validate it, and log the outcome either way.

    `transport` is a callable(prompt) -> str, injectable so tests can feed a
    canned reply without touching the network.
    """
    prompt = build_prompt(current_policy, candidate_outcomes, counterexample, focus=focus)
    source = "injected" if transport else mode()

    if transport is not None:
        raw = transport(prompt)
    elif source == "openai":
        raw = _call_openai(prompt)
    elif source == "ollama":
        raw = _call_ollama(prompt)
    else:
        raw = _call_mock(prompt, current_policy)

    record = {"source": source, "raw": raw[:4000]}
    try:
        patch = parse_patch(raw)
        new_policy = apply_patch(current_policy, patch)
    except (PatchRejected, policy_mod.PolicyValidationError) as exc:
        record.update({"accepted": False, "error": str(exc)})
        _log_call(record, log_path)
        return PatchOutcome(None, None, raw, source, error=str(exc))

    record.update({"accepted": True, "patch": patch, "policy": new_policy})
    _log_call(record, log_path)
    return PatchOutcome(patch, new_policy, raw, source)

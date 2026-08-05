"""The four non-negotiable constraints, as executable checks.

These are re-run at the end of every phase, so they live in one file rather than
being scattered through the tests for the phase that happened to introduce them.
Each test names the constraint it enforces and mirrors the grep from
refold_tasks.md as closely as a test can.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.agent import policy as policy_mod

SCANNED_TREES = ("src", "app", "docs")
AGENT_TREE = Path("src/agent")


def _files(*trees, suffixes=(".py", ".md", ".yaml", ".json")):
    out = []
    for tree in trees:
        root = Path(tree)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                out.append(path)
    return out


# --------------------------------------------------------------------------- #
# C1 — scope lock
# --------------------------------------------------------------------------- #

C1_PATTERN = re.compile(
    r"jepa|pepcompass|jacobian|therapeutic|restores function|discovers",
    re.IGNORECASE,
)


def _is_exclusion_context(path: Path, lines: list[str], index: int) -> bool:
    """True when the hit sits in a comment/caption that explains the exclusion.

    Accepts a comment line, a line inside a docstring-style block that negates
    the claim, or a nearby EXCLUSION marker — never bare shipped logic or pitch
    text that asserts the thing.
    """
    line = lines[index]
    stripped = line.strip()

    is_comment = stripped.startswith(("#", "//", "*")) or path.suffix in (".yaml",)
    negated = any(
        marker in line.lower()
        for marker in (
            "exclusion",
            "not an implementation",
            "never claim",
            "never claims",
            "nothing here",
            "does not",
            "do not",
            "must not",
            "no claim",
            "not implement",
            "never implement",
            "is not ",
            "no tangent",
            "forbidden",
        )
    )
    window = "\n".join(lines[max(0, index - 4) : index + 3]).lower()
    nearby_marker = "exclusion" in window or "constraint c1" in window or "c1" in window
    return is_comment or negated or nearby_marker


def test_c1_every_forbidden_term_sits_in_an_exclusion_context():
    """C1: no scope-creep term may appear in shipped logic or pitch text."""
    offences = []
    for path in _files(*SCANNED_TREES):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if C1_PATTERN.search(line) and not _is_exclusion_context(path, lines, index):
                offences.append(f"{path}:{index + 1}: {line.strip()}")
    assert not offences, "C1 violations:\n" + "\n".join(offences)


def test_c1_no_module_named_after_an_excluded_technique():
    for path in _files(*SCANNED_TREES, suffixes=(".py",)):
        assert not C1_PATTERN.search(path.stem), path


def test_c1_the_policy_dsl_cannot_express_an_excluded_computation():
    """Scope lock is structural, not only textual: the DSL has four features."""
    schema = policy_mod.load_schema()
    allowed = schema["properties"]["position_score"]["propertyNames"]["enum"]
    assert len(allowed) == 4
    assert not any(C1_PATTERN.search(name) for name in allowed)


# --------------------------------------------------------------------------- #
# C2 — the outer loop never executes code
# --------------------------------------------------------------------------- #

# The P3.2 grep, but with a leading guard so that a *method* call like
# `model.eval()` (putting a torch module in inference mode) is not mistaken for
# the builtin `eval()`. Bare `eval(`/`exec(` remain forbidden.
C2_PATTERN = re.compile(r"(?<![.\w])(eval|exec)\(|__import__|(?<![.\w])subprocess|os\.system")

# Files allowed to contain these tokens, with the reason.
C2_ALLOWLIST = {
    # spawns make_corruptions.py as a subprocess; no model or policy input reaches it
    Path("scripts/make_holdout.py"),
}


def test_c2_no_dynamic_execution_anywhere_on_the_policy_path():
    """C2: the interpreter and everything it imports are execution-free."""
    offenders = []
    for path in _files("src", "app", suffixes=(".py",)):
        if path in C2_ALLOWLIST:
            continue
        if C2_PATTERN.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert not offenders, f"dynamic execution found in: {offenders}"


@pytest.mark.parametrize(
    "payload",
    [
        {"exec": "os.system('rm -rf /')"},
        {"position_score": {"eval(": 1.0}},
        {"position_score": {"esm_surprisal": 1.0}, "proposal": {"import os": True}},
        {"position_score": {"__import__": 1.0}},
    ],
)
def test_c2_code_shaped_payloads_are_rejected_by_schema_validation(payload):
    """C2 verify: rejected by validation, not executed."""
    with pytest.raises(policy_mod.PolicyValidationError):
        policy_mod.validate_policy(payload)


def test_c2_the_interpreter_is_the_only_thing_that_consumes_a_policy():
    """No module builds behaviour out of policy text; it only reads named fields."""
    interpreter = Path("src/agent/policy_interpreter.py").read_text(encoding="utf-8")
    assert not C2_PATTERN.search(interpreter)
    # Field access is by literal key, never by a name taken from the payload.
    assert 'policy["position_score"]' in interpreter
    assert 'policy["proposal"]' in interpreter
    assert "getattr(policy" not in interpreter


# --------------------------------------------------------------------------- #
# C3 — the reference structure never reaches the agent
# --------------------------------------------------------------------------- #

def test_c3_agent_tree_never_mentions_the_reference():
    """C3 verify: grep for 'native' and 'SEQRES' under src/agent/ finds nothing."""
    for path in _files(str(AGENT_TREE)):
        text = path.read_text(encoding="utf-8")
        assert "native" not in text.lower(), f"{path} mentions the reference"
        assert "SEQRES" not in text, f"{path} mentions SEQRES"


def test_c3_only_the_evaluator_names_the_reference_path():
    hits = [
        p.as_posix()
        for p in Path("src").rglob("*.py")
        if "native_pdb_path" in p.read_text(encoding="utf-8")
    ]
    assert hits == ["src/evaluator.py"]


def test_c3_agent_tree_does_not_import_the_evaluator():
    pattern = re.compile(r"from\s+\.\.evaluator|from\s+src\.evaluator|import\s+evaluator")
    for path in _files(str(AGENT_TREE), suffixes=(".py",)):
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_c3_agent_tree_never_reads_the_protected_data_directory():
    for path in _files(str(AGENT_TREE), suffixes=(".py",)):
        text = path.read_text(encoding="utf-8")
        assert "data/proteins" not in text, path
        assert "PROTEINS" not in text, path
        assert "protein_dir" not in text, path


def test_c3_reveal_is_never_requested_from_agent_or_loop_code():
    for path in _files(str(AGENT_TREE), suffixes=(".py",)):
        assert "reveal=True" not in path.read_text(encoding="utf-8"), path


def test_c3_reveal_true_is_only_ever_passed_by_the_reveal_script():
    """Mentions in prose are fine; an actual call site is not.

    src/evaluator.py documents the parameter it defines, so a bare textual grep
    hits it. This looks for the argument in code, ignoring comments and the
    docstring lines that explain the rule.
    """
    call_sites = []
    for path in list(Path("src").rglob("*.py")) + list(Path("scripts").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "reveal=True" not in line:
                continue
            stripped = line.strip()
            is_prose = stripped.startswith("#") or "`reveal=True`" in stripped
            if not is_prose:
                call_sites.append(f"{path.as_posix()}:{number}")

    assert all(
        site.startswith("scripts/research/final_reveal.py") for site in call_sites
    ), call_sites
    assert call_sites, "the reveal script must actually call reveal=True"


# --------------------------------------------------------------------------- #
# C4 — no live inference on the judged path
# --------------------------------------------------------------------------- #

def test_c4_the_fold_cache_has_an_offline_switch():
    from src.cache import fold_cache

    assert hasattr(fold_cache, "FoldUnavailable")
    assert fold_cache.OFFLINE_ENV == "REFOLD_OFFLINE"


def test_c4_the_score_cache_has_an_offline_switch():
    from src.agent import esm_score

    assert hasattr(esm_score, "ScoringUnavailable")
    assert esm_score.OFFLINE_ENV == "REFOLD_OFFLINE"


def test_c4_offline_flag_is_read_at_call_time_not_import_time(monkeypatch):
    """A module-level constant would make the switch useless after import."""
    from src.cache import fold_cache

    monkeypatch.delenv("REFOLD_OFFLINE", raising=False)
    assert fold_cache.offline() is False
    monkeypatch.setenv("REFOLD_OFFLINE", "1")
    assert fold_cache.offline() is True
    monkeypatch.setenv("REFOLD_OFFLINE", "0")
    assert fold_cache.offline() is False


def test_c4_the_demo_driver_can_assert_zero_misses():
    from src import demo

    import inspect

    signature = inspect.signature(demo.run)
    assert "assert_no_misses" in signature.parameters
    assert hasattr(demo, "CacheMiss")


def test_c4_the_ui_reports_the_inference_mode():
    text = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "CACHED_BADGE" in text
    assert "REFOLD_OFFLINE" in text

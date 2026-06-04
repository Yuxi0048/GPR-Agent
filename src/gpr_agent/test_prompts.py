"""Deterministic guard: the 4 system prompts load and encode their rubric discipline.

This is the cheapest review-framework check -- it ties each A4 rubric dimension to a
required phrase in the relevant prompt, so a careless edit that drops "falsification"
or "not a calibrated probability" fails CI instead of silently weakening the agent.
Runs without pydantic-ai (prompts are plain markdown). `python -m ... test_prompts`
or pytest both work.
"""
from __future__ import annotations

from . import prompts

# agent key -> phrases (case-insensitive) that MUST appear (the discipline contract).
_REQUIRED = {
    "a1": ["opaque", "calibrated", "provenance", "do not interpret", "site context"],
    "a2": ["supporting_tools", "relative rank", "stay in your domain", "boundary", "uncertain prior"],
    "a3": ["migration-artefact", "relative rank", "stay in your domain", "uncertain prior"],
    "a4": [
        "falsif",                 # test by falsification, not confirmation
        "refuting",               # refuting_evidence sought actively
        "double count",           # no double counting across domains
        "one physical target = one hypothesis",
        "is_tentative",           # overclaim guard stays True
        "calibrated probability", # confidence is a relative rank, NOT a calibrated probability
        "circular",               # anti-circular: queries are questions, not conclusions
        "material library",       # no invented material names
        "site context",           # biases, never proves
    ],
}


def test_all_prompts_load_nonempty():
    for key in _REQUIRED:
        text = prompts.load(key)
        assert text.strip(), f"{key} prompt is empty"
        assert "DRAFT" in text, f"{key} should be marked DRAFT while under design"


def test_prompts_encode_rubric_discipline():
    for key, needles in _REQUIRED.items():
        text = prompts.load(key).lower()
        missing = [n for n in needles if n.lower() not in text]
        assert not missing, f"{key} prompt missing discipline markers: {missing}"


if __name__ == "__main__":
    test_all_prompts_load_nonempty()
    test_prompts_encode_rubric_discipline()
    for k in _REQUIRED:
        print(f"{k}: OK ({len(prompts.load(k))} chars, {len(_REQUIRED[k])} markers)")
    print("all prompt guards passed")

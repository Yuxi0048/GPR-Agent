"""LLM-as-judge reviewer for the PROCESS dimensions deterministic checks can't see.

The judge evaluates the agent's *process* against a rubric question -- reference-free
and GROUND-TRUTH-FREE -- so it can drive prompt + architecture refinement WITHOUT GT
leakage (the safe lever; see AGENTS_DESIGN first principle and the eval-protocol notes).
It does NOT score detections against survey GT; it asks "did the agent reason
honestly?" (overclaim, circularity, plausibility leaps, hidden ambiguity).

Controls (LLM-as-judge hygiene -- Zheng et al. 2023; Liu et al. G-Eval 2023):
  - Use a SEPARATE judge model from the agent (avoid self-preference bias) -- set via
    GPR_JUDGE_MODEL / `model=`; defaults to a different family than the agent default.
  - The judge is told to judge ONLY what is shown, not the true answer (no GT).
  - Verdicts are advisory until META-EVALUATED: calibrate the judge against a handful
    of human labels (agreement / kappa) before trusting it. Do not treat as truth.

Needs pydantic-ai + an API key. `run_llm_reviews(artifacts, model=...)`.
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent

from .review import REVIEW_BANK, Artifacts, ReviewResult


# Default judge model -- intentionally a DIFFERENT family than the agent default
# (agents default to anthropic) to reduce self-preference bias.
JUDGE_MODEL = os.environ.get("GPR_JUDGE_MODEL", "openai-chat:gpt-4o")

_JUDGE_PROMPT = """You are a strict, impartial reviewer of a GPR-interpretation agent's
PROCESS -- NOT its ground-truth accuracy. You are given the agent's evidence and its
interpretation, plus ONE review question about reasoning quality. Judge ONLY what is
shown; you do NOT know the true subsurface and must not guess it. Pass the agent only
if the artifacts genuinely satisfy the question. Watch for: overclaiming certainty or
a calibrated probability when none is earned; circular reasoning (a hypothesis or a
query asserting its own conclusion); physically implausible leaps; ambiguity that was
silently dropped instead of surfaced. Return passed + severity (ok/minor/major) + a
one-sentence rationale citing the specific artifact."""


class JudgeVerdict(BaseModel):
    passed: bool
    severity: Literal["ok", "minor", "major"]
    rationale: str


_judge = Agent(JUDGE_MODEL, output_type=JudgeVerdict, system_prompt=_JUDGE_PROMPT)


def _context(a: Artifacts) -> str:
    parts = []
    if a.imaging is not None:
        parts.append(f"IMAGING: velocity={a.imaging.velocity_m_per_ns} m/ns "
                     f"calibrated={a.imaging.velocity_calibrated}; note={a.imaging.provenance.notes}")
    for e in a.evidence:
        parts.append(f"EVIDENCE[{e.domain}] quality={e.quality!r} :: " + e.model_dump_json())
    if a.case is not None:
        parts.append("INTERPRETATION :: " + a.case.model_dump_json())
    return "\n".join(parts)


def judge_one(a: Artifacts, question, *, model=None) -> ReviewResult:
    """Run the judge on ONE review question over the artifacts (one API call)."""
    prompt = (f"Review question [{question.dimension.value}] ({question.target}):\n{question.question}\n\n"
              f"AGENT ARTIFACTS:\n{_context(a)}\n\n"
              f"Does the agent PASS this question? Judge the PROCESS, not the ground truth.")
    with _judge.override(model=model or JUDGE_MODEL):
        v = _judge.run_sync(prompt).output
    return ReviewResult(**question.model_dump(), passed=v.passed,
                        detail=f"[judge:{v.severity}] {v.rationale}")


def run_llm_reviews(a: Artifacts, *, model=None, bank=REVIEW_BANK) -> list[ReviewResult]:
    """Judge every `llm_judge` question in the bank (one API call each). GT-free."""
    return [judge_one(a, q, model=model) for q in bank if q.check == "llm_judge"]

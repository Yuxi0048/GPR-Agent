"""Review framework for the agents' inputs & outputs — an extensible question bank.

The A4 rubric (consistency · physical plausibility · calibrated uncertainty ·
no circular reasoning · no double-count · no overclaim · evidence-based)
operationalized as **review questions** over the typed handoffs. Each question
targets one artifact and has a check type:
  - ``deterministic`` : auto-checked by a function here (registered in CHECKERS);
  - ``llm_judge``     : posed to an evaluator LLM (an A4-style reviewer);
  - ``human``         : posed to a human reviewer.

HOW TO EXTEND (this is the point):
  1. add a ``ReviewQuestion(...)`` to ``REVIEW_BANK`` (any dimension/target);
  2. if it's auto-checkable, write ``def _check(a: Artifacts) -> (bool, str)`` and
     register it in ``CHECKERS[<id>]``;
  3. ``run_reviews(...)`` runs the deterministic ones and lists the rest as prompts.

DEV scaffold (runs on pydantic alone). The deterministic checks are heuristics —
sharpen them, or move questions deterministic <-> llm_judge <-> human, as you go.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Optional

from pydantic import BaseModel

from .contracts import DomainEvidence, HypothesisStatus, ImagingResult, InterpretationCase


class Dimension(str, Enum):
    consistency = "consistency"
    plausibility = "physical_plausibility"
    calibration = "calibrated_uncertainty"
    no_circular = "no_circular_reasoning"
    no_double_count = "no_double_count"
    no_overclaim = "no_overclaim"
    evidence_based = "evidence_based"


CheckType = Literal["deterministic", "llm_judge", "human"]
Target = Literal["A1.out", "A2.out", "A3.out", "A4.out", "A4.query"]


class ReviewQuestion(BaseModel):
    id: str
    dimension: Dimension
    target: Target
    question: str
    check: CheckType = "human"


@dataclass
class Artifacts:
    """The agent I/O under review (any subset)."""
    imaging: Optional[ImagingResult] = None
    evidence: list[DomainEvidence] = field(default_factory=list)
    case: Optional[InterpretationCase] = None


class ReviewResult(BaseModel):
    id: str
    dimension: Dimension
    target: Target
    question: str
    check: CheckType
    passed: Optional[bool] = None       # None = not auto-checkable (pose to llm/human)
    detail: str = ""


# --------------------------------------------------------------------------- #
# The starter question bank — append your own.
# --------------------------------------------------------------------------- #
REVIEW_BANK: list[ReviewQuestion] = [
    ReviewQuestion(id="EV-001", dimension=Dimension.evidence_based, target="A2.out", check="deterministic",
                   question="Does every detection cite >=1 supporting tool (supporting_tools)?"),
    ReviewQuestion(id="EV-002", dimension=Dimension.evidence_based, target="A3.out", check="deterministic",
                   question="Does every detection cite >=1 supporting tool (supporting_tools)?"),
    ReviewQuestion(id="OC-001", dimension=Dimension.no_overclaim, target="A4.out", check="deterministic",
                   question="Is is_tentative True and no hypothesis worded as a 'confirmed utility'/'guaranteed'?"),
    ReviewQuestion(id="CIRC-001", dimension=Dimension.no_circular, target="A4.query", check="deterministic",
                   question="Does any Query state a conclusion ('there is a pipe at R') instead of asking?"),
    ReviewQuestion(id="PLAUS-001", dimension=Dimension.plausibility, target="A4.out", check="deterministic",
                   question="Are velocity (0.03-0.3 m/ns) and depths (>=0) within physical bounds?"),
    ReviewQuestion(id="CAL-001", dimension=Dimension.calibration, target="A4.out", check="deterministic",
                   question="Are confidences treated as RELATIVE while velocity_calibrated is False?"),
    ReviewQuestion(id="DUP-001", dimension=Dimension.no_double_count, target="A4.out", check="deterministic",
                   question="Is each physical target ONE hypothesis (no detection id reused across hypotheses)?"),
    ReviewQuestion(id="CONS-001", dimension=Dimension.consistency, target="A4.out", check="deterministic",
                   question="For each cross-domain-confirmed hypothesis, do the cited A2 + A3 detections agree in (x, depth)?"),
    ReviewQuestion(id="EV-003", dimension=Dimension.evidence_based, target="A4.out", check="deterministic",
                   question="Does every non-refuted, non-clutter hypothesis trace to >=1 detection id?"),
    ReviewQuestion(id="CIRC-002", dimension=Dimension.no_circular, target="A4.out", check="human",
                   question="Was site context used only as an uncertain prior, never as ground truth?"),
    # --- genuinely non-deterministic: the LLM judge evaluates the PROCESS (reference-free,
    #     GT-free), so it can drive prompt refinement WITHOUT ground-truth leakage. ---
    ReviewQuestion(id="NARR-001", dimension=Dimension.no_overclaim, target="A4.out", check="llm_judge",
                   question="Does `overall_note` convey the uncertainty honestly (tentative, relative confidence) "
                            "without language that implies certainty or a calibrated probability?"),
    ReviewQuestion(id="PLAUS-002", dimension=Dimension.plausibility, target="A4.out", check="llm_judge",
                   question="Is each hypothesis's stated kind/depth physically consistent with its cited evidence "
                            "(signature, depth band, cross-domain agreement) -- no physically implausible leaps?"),
    ReviewQuestion(id="HON-001", dimension=Dimension.evidence_based, target="A2.out", check="llm_judge",
                   question="Are low-SNR / ambiguous / boundary regions honestly surfaced in `quality`, rather than "
                            "silently dropped, given the detections reported?"),
    ReviewQuestion(id="CIRC-003", dimension=Dimension.no_circular, target="A4.out", check="llm_judge",
                   question="Is the reasoning free of circular logic -- no hypothesis used as its own evidence, "
                            "and follow-up queries phrased as open questions rather than asserted conclusions?"),
]


# --------------------------------------------------------------------------- #
# Deterministic checkers (id -> fn(Artifacts) -> (passed, detail))
# --------------------------------------------------------------------------- #
_BANNED = ("confirmed utility", "guaranteed", "definitely", "certainly a", "100%")
_CONCLUSION_TELLS = ("there is a", "this is a", "we found a", "confirmed")


def _evidence_cited(a: Artifacts, domain: str):
    ev = [e for e in a.evidence if e.domain == domain]
    if not ev:
        return None, f"no {domain} evidence supplied"
    bad = [d.id for e in ev for d in e.detections if not d.supporting_tools]
    return (not bad), ("ok" if not bad else f"detections without supporting_tools: {bad}")


def _check_ev001(a): return _evidence_cited(a, "radargram")
def _check_ev002(a): return _evidence_cited(a, "subsurface")


def _check_oc001(a):
    if a.case is None:
        return None, "no InterpretationCase"
    bad = [h.id for h in a.case.hypotheses if any(b in h.statement.lower() for b in _BANNED)]
    ok = a.case.is_tentative and not bad
    return ok, ("ok" if ok else f"is_tentative={a.case.is_tentative}; overclaiming={bad}")


def _check_circ001(a):
    if a.case is None:
        return None, "no InterpretationCase"
    bad = [q.question for q in a.case.follow_up_queries
           if any(t in q.question.lower() for t in _CONCLUSION_TELLS)]
    return (not bad), ("ok" if not bad else f"queries stating conclusions: {bad}")


def _check_plaus001(a):
    if a.case is None:
        return None, "no InterpretationCase"
    issues = []
    if a.imaging and a.imaging.velocity_m_per_ns is not None:
        v = a.imaging.velocity_m_per_ns
        if not (0.03 <= v <= 0.3):
            issues.append(f"velocity {v} m/ns out of [0.03,0.3]")
    for e in a.evidence:
        for d in e.detections:
            if d.depth_m is not None and d.depth_m < 0:
                issues.append(f"{d.id}: negative depth {d.depth_m}")
    return (not issues), ("ok" if not issues else "; ".join(issues))


def _check_cal001(a):
    if a.case is None:
        return None, "no InterpretationCase"
    calibrated = bool(a.imaging and a.imaging.velocity_calibrated)
    # If not calibrated, any confidence == 1.0 is an overclaim of certainty.
    overcertain = [h.id for h in a.case.hypotheses if h.confidence >= 1.0]
    ok = calibrated or not overcertain
    return ok, ("calibrated" if calibrated else
                ("ok (no 1.0 confidences)" if ok else f"uncalibrated but conf=1.0: {overcertain}"))


# Tolerances for the cross-domain consistency check (match deterministic.FROZEN_PARAMS;
# duplicated here to keep review.py pydantic-only -- no numpy/engine import).
_DX_TOL, _DZ_TOL = 0.4, 0.15


def _detections_by_id(a: Artifacts) -> dict:
    return {d.id: d for e in a.evidence for d in e.detections}


def _check_dup001(a):
    """No double count: a detection id must not back more than one hypothesis."""
    if a.case is None:
        return None, "no InterpretationCase"
    seen, dup = set(), []
    for h in a.case.hypotheses:
        for eid in h.supporting_evidence:
            (dup.append(eid) if eid in seen else seen.add(eid))
    return (not dup), ("ok (evidence partitioned across hypotheses)" if not dup
                       else f"detection(s) reused across hypotheses (double count): {sorted(set(dup))}")


def _check_cons001(a):
    """Consistency: each cross-domain-confirmed hypothesis must cite a radargram AND a
    subsurface detection that agree in (x, depth) within tolerance."""
    if a.case is None:
        return None, "no InterpretationCase"
    dets = _detections_by_id(a)
    issues = []
    for h in a.case.hypotheses:
        if not h.cross_domain_confirmed:
            continue
        cited = [dets[e] for e in h.supporting_evidence if e in dets]
        r = [d for d in cited if d.domain == "radargram"]
        s = [d for d in cited if d.domain == "subsurface"]
        if not (r and s):
            issues.append(f"{h.id}: confirmed but missing both-domain evidence")
            continue
        if r[0].x_m is not None and s[0].x_m is not None and abs(r[0].x_m - s[0].x_m) > _DX_TOL:
            issues.append(f"{h.id}: A2/A3 x disagree ({r[0].x_m:.2f} vs {s[0].x_m:.2f})")
        if r[0].depth_m is not None and s[0].depth_m is not None and abs(r[0].depth_m - s[0].depth_m) > _DZ_TOL:
            issues.append(f"{h.id}: A2/A3 depth disagree ({r[0].depth_m:.2f} vs {s[0].depth_m:.2f})")
    return (not issues), ("ok (confirmed hyps cite agreeing A2+A3 evidence)" if not issues else "; ".join(issues))


def _check_ev003(a):
    """Evidence-based: every non-refuted, non-clutter hypothesis cites >=1 detection id."""
    if a.case is None:
        return None, "no InterpretationCase"
    bad = [h.id for h in a.case.hypotheses
           if h.status != HypothesisStatus.refuted
           and "clutter" not in h.statement.lower()
           and not h.supporting_evidence]
    return (not bad), ("ok (every claim traces to evidence)" if not bad
                       else f"hypotheses asserted without a detection id: {bad}")


CHECKERS: dict[str, Callable[[Artifacts], tuple[Optional[bool], str]]] = {
    "EV-001": _check_ev001, "EV-002": _check_ev002, "OC-001": _check_oc001,
    "CIRC-001": _check_circ001, "PLAUS-001": _check_plaus001, "CAL-001": _check_cal001,
    "DUP-001": _check_dup001, "CONS-001": _check_cons001, "EV-003": _check_ev003,
}


def run_reviews(artifacts: Artifacts, bank: list[ReviewQuestion] = REVIEW_BANK) -> list[ReviewResult]:
    """Run deterministic checks; surface llm_judge/human questions as prompts."""
    out: list[ReviewResult] = []
    for q in bank:
        passed, detail = None, ""
        if q.check == "deterministic" and q.id in CHECKERS:
            passed, detail = CHECKERS[q.id](artifacts)
        elif q.check != "deterministic":
            detail = f"pose to {q.check}"
        out.append(ReviewResult(**q.model_dump(), passed=passed, detail=detail))
    return out


def summary(results: list[ReviewResult]) -> dict:
    det = [r for r in results if r.check == "deterministic"]
    return {
        "auto_passed": sum(1 for r in det if r.passed is True),
        "auto_failed": sum(1 for r in det if r.passed is False),
        "auto_inapplicable": sum(1 for r in det if r.passed is None),
        "to_llm_judge": sum(1 for r in results if r.check == "llm_judge"),
        "to_human": sum(1 for r in results if r.check == "human"),
    }

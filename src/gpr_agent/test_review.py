"""The promoted deterministic checks (DUP-001 / CONS-001 / EV-003) must CATCH
violations, not just pass clean cases. Pure pydantic -- no engine, no API.
"""
from __future__ import annotations

from . import review
from .contracts import (
    Detection, DomainEvidence, Hypothesis, HypothesisStatus, InterpretationCase, Provenance,
)


def _prov(agent="A4"):
    return Provenance(agent=agent)


def _evidence(r_xz=(2.0, 1.0), s_xz=(2.0, 1.0)):
    r = Detection(id="r000", domain="radargram", x_m=r_xz[0], depth_m=r_xz[1], kind="apex",
                  supporting_tools=["envelope"])
    s = Detection(id="s000", domain="subsurface", x_m=s_xz[0], depth_m=s_xz[1], kind="blob",
                  supporting_tools=["maximum_filter"])
    return [DomainEvidence(domain="radargram", detections=[r], provenance=_prov("A2")),
            DomainEvidence(domain="subsurface", detections=[s], provenance=_prov("A3"))]


def _arts(hyps, evidence=None):
    return review.Artifacts(imaging=None, evidence=evidence or _evidence(),
                            case=InterpretationCase(hypotheses=hyps, provenance=_prov()))


def _confirmed(hid="H001", ev=("r000", "s000")):
    return Hypothesis(id=hid, statement="point reflector", status=HypothesisStatus.supported,
                      x_m=2.0, depth_m=1.0, supporting_evidence=list(ev), cross_domain_confirmed=True)


def test_clean_case_passes_all_three():
    a = _arts([_confirmed()])
    for cid in ("DUP-001", "CONS-001", "EV-003"):
        passed, detail = review.CHECKERS[cid](a)
        assert passed is True, f"{cid} should pass clean case: {detail}"


def test_dup001_catches_reused_detection():
    # two hypotheses both cite r000 -> the same detection backs two targets (double count)
    a = _arts([_confirmed("H001"), Hypothesis(id="H002", statement="another",
              status=HypothesisStatus.ambiguous, x_m=2.0, depth_m=1.0, supporting_evidence=["r000"])])
    passed, detail = review.CHECKERS["DUP-001"](a)
    assert passed is False and "r000" in detail


def test_cons001_catches_cross_domain_disagreement():
    # confirmed hypothesis whose cited A2 (depth 1.0) and A3 (depth 1.6) disagree beyond tol
    a = _arts([_confirmed()], evidence=_evidence(r_xz=(2.0, 1.0), s_xz=(2.0, 1.6)))
    passed, detail = review.CHECKERS["CONS-001"](a)
    assert passed is False and "depth disagree" in detail


def test_cons001_catches_missing_domain():
    # 'confirmed' but only cites a radargram detection -> not actually cross-domain
    a = _arts([_confirmed(ev=("r000",))])
    passed, detail = review.CHECKERS["CONS-001"](a)
    assert passed is False and "both-domain" in detail


def test_ev003_catches_unsupported_hypothesis():
    a = _arts([Hypothesis(id="H001", statement="a pipe", status=HypothesisStatus.supported,
                          x_m=2.0, depth_m=1.0, supporting_evidence=[])])
    passed, detail = review.CHECKERS["EV-003"](a)
    assert passed is False and "H001" in detail


def test_ev003_allows_refuted_and_clutter_without_evidence():
    a = _arts([Hypothesis(id="H001", statement="all clutter", status=HypothesisStatus.refuted,
                          supporting_evidence=[])])
    passed, _ = review.CHECKERS["EV-003"](a)
    assert passed is True

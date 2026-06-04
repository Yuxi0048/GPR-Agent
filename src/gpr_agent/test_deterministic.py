"""CI guard for the keyless A1->(A2||A3)->A4 loop on a tiny synthetic radargram.

No gpr_bench, no GT, no key -- just the shared tools + deterministic A4 over a few
planted point reflectors. Asserts the contract discipline holds end-to-end: typed
output, evidence-cited detections, relative (sub-1.0) confidences, the review rubric
passing, and cross-domain de-dup producing one hypothesis per target from BOTH
domains. Skips if gpr_data_processing (the envelope/dip engine) isn't importable.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gpr_data_processing", reason="needs the gpr-data-processing engine")

from . import review                                  # noqa: E402
from .contracts import HypothesisStatus, InterpretationCase  # noqa: E402
from .deterministic import orchestrate_deterministic  # noqa: E402
from .session import Session                           # noqa: E402


def _synthetic(n_s: int = 256, n_t: int = 200, seed: int = 0) -> np.ndarray:
    """A few smooth, bright point reflectors (sample, trace) over mild noise."""
    rng = np.random.default_rng(seed)
    img = 0.02 * rng.standard_normal((n_s, n_t))
    rows, cols = np.ogrid[:n_s, :n_t]
    for s, t, amp in [(60, 30, 1.0), (120, 100, 0.9), (200, 150, 0.8)]:
        img += amp * np.exp(-((rows - s) ** 2 / (2 * 4.0 ** 2) + (cols - t) ** 2 / (2 * 6.0 ** 2)))
    return img


def _run():
    s = Session(arrays={"raw": _synthetic()})
    # migrate=False: the cheap depth-relabelled path -- this test guards the contract
    # discipline (typed, evidence-cited, de-dup, rubric), not the migration numerics.
    return orchestrate_deterministic(s, "raw", dt_ns=0.1, dx_m=0.02, max_rounds=2, migrate=False)


def test_loop_emits_a_valid_interpretation_case():
    res = _run()
    assert isinstance(res.case, InterpretationCase)
    assert res.case.is_tentative is True
    assert res.case.hypotheses, "expected >=1 hypothesis from planted reflectors"


def test_every_detection_is_evidence_cited_and_relative():
    res = _run()
    for ev in res.evidence:
        assert ev.detections, f"{ev.domain} produced no detections"
        for d in ev.detections:
            assert d.supporting_tools, f"{d.id} has no supporting_tools (not evidence-based)"
            assert d.confidence < 1.0, f"{d.id} confidence {d.confidence} must be a relative rank (<1.0)"


def test_cross_domain_dedup_makes_one_hypothesis_from_both_domains():
    res = _run()
    confirmed = [h for h in res.case.hypotheses if h.cross_domain_confirmed]
    assert confirmed, "planted reflectors should confirm across radargram+subsurface"
    for h in confirmed:
        assert h.status == HypothesisStatus.supported
        domains = {e[0] for e in h.supporting_evidence}   # 'r...' vs 's...'
        assert domains == {"r", "s"}, f"{h.id} should fuse ONE apex + ONE blob, got {h.supporting_evidence}"


def test_review_rubric_passes_no_deterministic_failures():
    res = _run()
    results = review.run_reviews(review.Artifacts(imaging=res.imaging, evidence=res.evidence, case=res.case))
    failures = [r.id for r in results if r.passed is False]
    assert not failures, f"deterministic review checks failed: {failures}"


def test_follow_up_queries_are_questions_not_conclusions():
    res = _run()
    for q in res.case.follow_up_queries:
        assert q.question.strip().endswith("?"), f"re-plan query must be a question: {q.question!r}"


def test_a1_migration_path_produces_a_distinct_section():
    """A1's real Kirchhoff migration runs and yields a subsurface section that is NOT
    just the B-scan (A3's input genuinely differs). Tiny array; skips without pylops."""
    pytest.importorskip("pylops", reason="migration needs the [migration] extra")
    from . import tools
    s = Session(arrays={"raw": _synthetic(n_s=64, n_t=48, seed=1)})
    img = tools.image(s, "raw", dt_ns=0.2, dx_m=0.05, migrate="adjoint", bgr=True, nz=48)
    assert img.subsurface.handle == "subsurface"
    assert any(t.tool == "migrate_pylops" for t in img.provenance.tools), "migration not recorded"
    assert s.arrays["subsurface"].shape != s.arrays["bscan"].shape or \
        not np.allclose(s.arrays["subsurface"], s.arrays["bscan"]), "subsurface == B-scan (not migrated)"
    assert img.velocity_calibrated is False   # nominal velocity stays honest

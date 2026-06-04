"""Keyless deterministic A1 -> (A2 || A3) -> A4 loop, scored vs GPR-Bench + reviewed.

The full agent topology with NO LLM and NO API key: the shared `tools.py` detectors
feed a deterministic geometric A4 (`deterministic.fuse`) that does cross-domain
association -> de-dup -> falsification -> bounded re-plan, emitting a real
`InterpretationCase`. We then (a) score the A4 case against the dataset GT via
`gpr_bench.eval`, alongside the simpler baselines from `run_agent_validation`, and
(b) run the `review` framework over the artifacts (the A4 rubric, auto-checked).

This is the deterministic ORACLE the VLM agents are compared against -- a real, if
geometric, end-to-end run that proves the skeleton before any VLM spend.

Run (after `make setup`, or GPR_TOOLS_SRC + PYTHONPATH=<GPR-Bench>):

    python run_deterministic_loop.py [dataset]        # default: tu1208
"""
from __future__ import annotations

import sys

import run_agent_validation as base          # reuse baseline + A2 rows for comparison
from gpr_agent import review
from gpr_agent.contracts import HypothesisStatus
from gpr_agent.deterministic import orchestrate_deterministic
from gpr_agent.session import Session


def _case_to_dets(case, statuses):
    from gpr_bench.eval import Detection
    return [Detection(x_m=h.x_m, depth_m=h.depth_m, conf=h.confidence)
            for h in case.hypotheses if h.status in statuses and h.depth_m is not None]


def main(dataset: str = "tu1208", max_rounds: int = 2) -> dict:
    from gpr_bench import eval as gpr_eval
    from gpr_bench.datasets.registry import get_adapter

    ad = get_adapter(dataset)
    line = ad.lines()[0]
    rg = ad.load(line)
    targets = [t for t in ad.targets(line) if t.get("depth_m") is not None]
    print(f"[{ad.name}] line {rg.data.shape} (n_samp x n_tr); {len(targets)} GT targets with depth\n")

    # ---- the keyless loop ----
    s = Session(arrays={"raw": rg.data})
    res = orchestrate_deterministic(s, "raw", dt_ns=rg.dt_ns or 1.0, dx_m=rg.dx_m or 1.0,
                                    max_rounds=max_rounds)
    case = res.case

    # ---- score the A4 case vs GT, alongside the simpler stand-ins ----
    methods = {
        "baseline (envelope stack)":   base.perceive_baseline(rg),
        "A2 only (apex depths)":       base.perceive_a2_radargram(rg),
        "A4 case: confirmed only":     _case_to_dets(case, {HypothesisStatus.supported}),
        "A4 case: confirmed+ambiguous": _case_to_dets(case, {HypothesisStatus.supported,
                                                              HypothesisStatus.ambiguous}),
    }
    hdr = f"{'method':30s} {'n_det':>5} {'prec':>6} {'recall':>6} {'AP':>6} {'depthMAE':>9}"
    print(hdr); print("-" * len(hdr))
    reports = {}
    for name, dets in methods.items():
        r = gpr_eval.evaluate(dets, targets, mode="depth")
        reports[name] = r
        mae = r.get("localization", {}).get("depth_mae")
        mae_s = f"{mae:.3f}" if mae is not None else "-"
        print(f"{name:30s} {len(dets):5d} {r['precision']:6.2f} {r['recall']:6.2f} "
              f"{r['average_precision']:6.2f} {mae_s:>9}")

    # ---- the A4 InterpretationCase ----
    print(f"\nA4 case: {res.rounds} re-plan round(s) | {case.overall_note}")
    for h in case.hypotheses[:12]:
        loc = f"x~{h.x_m:.2f} z~{h.depth_m:.2f}" if h.x_m is not None else "-"
        flag = "[CONFIRMED]" if h.cross_domain_confirmed else ""
        print(f"  {h.id} {h.status.value:9s} conf={h.confidence:.2f} {loc:18s} "
              f"ev={h.supporting_evidence} {flag}")
    if case.follow_up_queries:
        print(f"  + {len(case.follow_up_queries)} unresolved follow-up QUESTION(s) (re-plan, not conclusions)")

    # ---- run the review framework over the artifacts (the A4 rubric) ----
    arts = review.Artifacts(imaging=res.imaging, evidence=res.evidence, case=case)
    results = review.run_reviews(arts)
    summ = review.summary(results)
    print(f"\nReview: auto {summ['auto_passed']} pass / {summ['auto_failed']} fail / "
          f"{summ['auto_inapplicable']} n-a; {summ['to_llm_judge']} -> llm-judge, {summ['to_human']} -> human")
    for r in results:
        if r.passed is False:
            print(f"  FAIL {r.id} [{r.dimension.value}] {r.detail}")
    print("\nLoop: GPR-Bench -> A1 image -> (A2 apex || A3 blob) -> A4 fuse/de-dup/re-plan "
          "-> InterpretationCase, scored (depth, dz_tol=0.15) + reviewed. No LLM, no key.")
    return {"reports": reports, "review": summ, "rounds": res.rounds,
            "n_hyp": len(case.hypotheses)}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tu1208")

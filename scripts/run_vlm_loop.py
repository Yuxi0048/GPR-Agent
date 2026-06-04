"""VLM agentic loop vs the deterministic oracle, on one GPR-Bench line.

Runs the SAME A1->(A2||A3)->A4 topology two ways and scores both against the GT:
  - oracle: `deterministic.orchestrate_deterministic` (shared tools, no LLM);
  - VLM:    `agents.orchestrate` -- A1 images deterministically, then A2/A3 SEE the
            rendered section + call the tools, and A4 fuses (real model).
Uses real API calls (ANTHROPIC_API_KEY / OPENAI_API_KEY); pick the model with
GPR_AGENT_MODEL or --model (e.g. anthropic:claude-sonnet-4-6, openai-chat:gpt-4o).

NO CALIBRATION (first principle): nothing is fit to GT, confidences are relative,
velocity is nominal -- every number here is a dev sanity bar, not a tuned result.

Run:
    PYTHONPATH=<GPR-Bench>;<GPR-Tools>/src  python run_vlm_loop.py [dataset] [model]
"""
from __future__ import annotations

import sys
import time

from gpr_agent import agents, review
from gpr_agent.contracts import HypothesisStatus
from gpr_agent.deterministic import orchestrate_deterministic
from gpr_agent.session import Session


def _case_to_dets(case, statuses):
    from gpr_bench.eval import Detection
    return [Detection(x_m=h.x_m, depth_m=h.depth_m, conf=h.confidence)
            for h in case.hypotheses if h.status in statuses and h.depth_m is not None]


def _report(label, res, targets):
    from gpr_bench import eval as gpr_eval

    print(f"\n=== {label} ===")
    sup = {HypothesisStatus.supported}
    both = {HypothesisStatus.supported, HypothesisStatus.ambiguous}
    for name, statuses in [("confirmed only", sup), ("confirmed+ambiguous", both)]:
        dets = _case_to_dets(res.case, statuses)
        r = gpr_eval.evaluate(dets, targets, mode="depth")
        mae = r.get("localization", {}).get("depth_mae")
        print(f"  {name:22s} n={len(dets):3d}  P={r['precision']:.2f} R={r['recall']:.2f} "
              f"AP={r['average_precision']:.2f} depthMAE={mae:.3f}" if mae is not None
              else f"  {name:22s} n={len(dets):3d}  P={r['precision']:.2f} R={r['recall']:.2f} AP={r['average_precision']:.2f}")
    n_conf = sum(h.cross_domain_confirmed for h in res.case.hypotheses)
    print(f"  hypotheses={len(res.case.hypotheses)} ({n_conf} cross-domain-confirmed); "
          f"is_tentative={res.case.is_tentative}; queries={len(res.case.follow_up_queries)}")
    rv = review.run_reviews(review.Artifacts(imaging=res.imaging, evidence=res.evidence, case=res.case))
    s = review.summary(rv)
    fails = [f"{r.id}:{r.detail}" for r in rv if r.passed is False]
    print(f"  review: {s['auto_passed']} pass / {s['auto_failed']} fail / {s['auto_inapplicable']} n-a"
          + (f"  FAILS={fails}" if fails else ""))
    if res.usage:
        print(f"  tokens: in={res.usage['input']} out={res.usage['output']} ({res.usage['requests']} requests)")


def main(dataset: str = "tu1208", model: str | None = None) -> dict:
    from gpr_bench.datasets.registry import get_adapter

    ad = get_adapter(dataset)
    line = ad.lines()[0]
    rg = ad.load(line)
    targets = [t for t in ad.targets(line) if t.get("depth_m") is not None]
    dt_ns, dx_m = rg.dt_ns or 1.0, rg.dx_m or 1.0
    print(f"[{ad.name}] line {rg.data.shape}; {len(targets)} GT targets w/ depth | model={model or agents.MODEL}")

    oracle = orchestrate_deterministic(Session(arrays={"raw": rg.data}), "raw",
                                       dt_ns=dt_ns, dx_m=dx_m, migrate="adjoint")
    _report("DETERMINISTIC ORACLE (no LLM)", oracle, targets)

    t = time.perf_counter()
    vlm = agents.orchestrate(Session(arrays={"raw": rg.data}), "raw",
                             dt_ns=dt_ns, dx_m=dx_m, migrate="adjoint", model=model)
    print(f"\n[VLM loop ran in {time.perf_counter()-t:.1f}s]")
    _report("VLM AGENTS (A2/A3 see the image + call tools; A4 fuses)", vlm, targets)

    print("\nSame topology, two engines. NO CALIBRATION -- relative confidences, nominal "
          "velocity, single line: a dev sanity bar, not a tuned/calibrated result.")
    return {"oracle_hyp": len(oracle.case.hypotheses), "vlm_hyp": len(vlm.case.hypotheses),
            "vlm_usage": vlm.usage}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a]
    ds = args[0] if args else "tu1208"
    md = args[1] if len(args) > 1 else None
    main(ds, md)

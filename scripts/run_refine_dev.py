"""Leakage-free VLM refinement loop on a DEV set (synthetic scenes).

For each dev scene: run the deterministic ORACLE and the VLM, then score the VLM by
GT-FREE signals only:
  1. PROCESS rubric -- 9 deterministic checks (review) + the LLM judge (review_llm);
  2. ORACLE-AGREEMENT -- does the VLM recover the oracle's detections? (silver, not GT);
  3. conservatism -- VLM hypothesis count vs the oracle.
The frozen TU1208/Purdue test set is NEVER used here. Read the judge FAIL rationales
+ oracle-agreement recall, edit prompts/*.md, re-run -- that is the refinement loop.
(A synthetic "planted-recall" line is printed as a DEV-ONLY sanity, not a tuning target.)

Run (uses API for the VLM + judge):
    PYTHONPATH=<GPR-Tools>/src  python run_refine_dev.py [n_scenes] [model]
"""
from __future__ import annotations

import sys

from gpr_agent import agents, review, review_llm
from gpr_agent.contracts import HypothesisStatus
from gpr_agent.deterministic import orchestrate_deterministic
from gpr_agent.dev_scenes import synth_scene
from gpr_agent.session import Session

_DX_TOL, _DZ_TOL = 0.4, 0.2


def _xz(case, statuses):
    return [(h.x_m, h.depth_m) for h in case.hypotheses
            if h.status in statuses and h.depth_m is not None]


def _agreement(pred, ref):
    """Greedy (x, depth) match of pred vs ref -> (precision, recall, tp)."""
    used, tp = set(), 0
    for px, pz in pred:
        for i, (rx, rz) in enumerate(ref):
            if i in used:
                continue
            if (px is None or rx is None or abs(px - rx) <= _DX_TOL) and abs(pz - rz) <= _DZ_TOL:
                used.add(i); tp += 1; break
    return (tp / len(pred) if pred else 0.0), (tp / len(ref) if ref else 0.0), tp


def _process(arts, judge_model):
    det = [r for r in review.run_reviews(arts) if r.check == "deterministic"]
    jud = review_llm.run_llm_reviews(arts, model=judge_model)
    det_fail = [f"{r.id}:{r.detail}" for r in det if r.passed is False]
    jud_fail = [f"{r.id}:{r.detail}" for r in jud if r.passed is False]
    return {
        "det_pass": sum(r.passed is True for r in det), "det_n": len(det),
        "jud_pass": sum(r.passed is True for r in jud), "jud_n": len(jud),
        "fails": det_fail + jud_fail,
    }


def main(n_scenes: int = 2, model: str | None = None, judge_model: str | None = None) -> dict:
    print(f"DEV refinement | {n_scenes} synthetic scene(s) | agent={model or agents.MODEL} "
          f"| judge={judge_model or review_llm.JUDGE_MODEL}")
    print("GT-free signals: process rubric (det+judge) + oracle-agreement. TU1208 NOT used.\n")
    agg = {"vlm_hyp": 0, "ora_hyp": 0, "det_pass": 0, "det_n": 0, "jud_pass": 0, "jud_n": 0,
           "oracle_R": 0.0, "planted_R": 0.0}
    for i in range(n_scenes):
        img, dt_ns, dx_m, planted = synth_scene(seed=i)
        keep = {HypothesisStatus.supported, HypothesisStatus.ambiguous}

        ora = orchestrate_deterministic(Session(arrays={"raw": img}), "raw", dt_ns=dt_ns, dx_m=dx_m, migrate="adjoint")
        vlm = agents.orchestrate(Session(arrays={"raw": img}), "raw", dt_ns=dt_ns, dx_m=dx_m, migrate="adjoint", model=model)

        ora_xz, vlm_xz = _xz(ora.case, keep), _xz(vlm.case, keep)
        _, oracle_R, otp = _agreement(vlm_xz, ora_xz)              # VLM recall vs oracle (silver)
        _, planted_R, ptp = _agreement(vlm_xz, planted)           # vs synthetic truth (dev sanity)
        proc = _process(review.Artifacts(imaging=vlm.imaging, evidence=vlm.evidence, case=vlm.case), judge_model)

        print(f"scene {i}: planted={len(planted)} | oracle hyps={len(ora_xz)} | VLM hyps={len(vlm_xz)}")
        print(f"  PROCESS : deterministic {proc['det_pass']}/{proc['det_n']} | "
              f"judge {proc['jud_pass']}/{proc['jud_n']}")
        for f in proc["fails"]:
            print(f"     FAIL {f[:110]}")
        print(f"  ORACLE-AGREEMENT (silver): VLM recovers {otp}/{len(ora_xz)} oracle dets (R={oracle_R:.2f})")
        print(f"  (dev sanity: VLM hits {ptp}/{len(planted)} planted targets, R={planted_R:.2f})\n")

        agg["vlm_hyp"] += len(vlm_xz); agg["ora_hyp"] += len(ora_xz)
        for kk in ("det_pass", "det_n", "jud_pass", "jud_n"):
            agg[kk] += proc[kk]
        agg["oracle_R"] += oracle_R; agg["planted_R"] += planted_R

    n = max(1, n_scenes)
    print(f"== DEV SUMMARY ({n_scenes} scenes) ==")
    print(f"process: deterministic {agg['det_pass']}/{agg['det_n']} | judge {agg['jud_pass']}/{agg['jud_n']}")
    print(f"oracle-agreement recall (mean): {agg['oracle_R']/n:.2f} | VLM/oracle hyp ratio: "
          f"{agg['vlm_hyp']}/{agg['ora_hyp']}")
    print(f"planted-recall (dev sanity, mean): {agg['planted_R']/n:.2f}")
    print("\nRefine: read judge FAIL rationales + raise oracle-agreement recall by editing "
          "prompts/*.md (e.g. pass-through tool detections). NEVER tune to TU1208.")
    return agg


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a]
    n = int(args[0]) if args else 2
    md = args[1] if len(args) > 1 else None
    main(n, md)

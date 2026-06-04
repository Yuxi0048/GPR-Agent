"""Run the AGENTIC (VLM) workflow on every scorable TU1208 line and report.

Mirror of `run_deterministic_all.py`, but the perception is the VLM agent loop
(`agents.orchestrate`): A1 images deterministically + renders, A2 SEES the B-scan and
A3 SEES the migrated section (each calls tools), then the hybrid A4 fuses (shared
geometry) and the VLM writes the narrative. Real API calls -- set OPENAI_API_KEY (or
ANTHROPIC_API_KEY) and the model via GPR_AGENT_MODEL / --model.

Same frozen, GT-free discipline as the oracle: nothing is tuned to TU1208, velocity is
nominal, confidences are relative -> a dev sanity bar, not a calibrated result. We score
the VLM case vs the per-region GT (depth, dz_tol=0.15) over the SAME 63 scorable lines the
oracle ran on, and print it side-by-side with the oracle numbers loaded from
`_artifacts/tu1208_all_deterministic.json`.

Results are CHECKPOINTED to `_artifacts/tu1208_all_vlm.json` after every line, so a
mid-sweep API failure/interrupt never loses completed work (re-run resumes by skipping
files already present unless --fresh).

Run:
    PYTHONPATH="<GPR-Bench>;<GPR-Tools>/src" GPR_AGENT_MODEL=openai-chat:gpt-4o \
        python run_vlm_all.py [db_root]
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from gpr_agent.contracts import HypothesisStatus
from gpr_agent.session import Session
from run_deterministic_all import ART, _aggregate, _case_to_dets, _metrics

MODEL = os.environ.get("GPR_AGENT_MODEL", "openai-chat:gpt-4o")
ORACLE_JSON = ART / "tu1208_all_deterministic.json"
VLM_JSON = ART / "tu1208_all_vlm.json"


def run_one(ad, line, gpr_eval, parse_name):
    fm = parse_name(Path(line).name)
    rg = ad.load(line)
    targets = [t for t in ad.targets(line) if t.get("depth_m") is not None]
    rec = {"file": Path(line).name, "region": fm.region, "line": fm.line,
           "freq_mhz": fm.freq_mhz, "vendor": fm.vendor,
           "n_samples": int(rg.data.shape[0]), "n_traces": int(rg.data.shape[1]),
           "dt_ns": rg.dt_ns, "dx_m": rg.dx_m, "n_gt": len(targets)}

    from gpr_agent import agents
    t0 = time.perf_counter()
    res = agents.orchestrate(Session(arrays={"raw": rg.data}), "raw",
                             dt_ns=rg.dt_ns or 1.0, dx_m=rg.dx_m or 1.0,
                             migrate="adjoint", model=MODEL)
    rec["secs"] = round(time.perf_counter() - t0, 1)
    rec["usage"] = res.usage
    case = res.case
    rec["n_hyp"] = len(case.hypotheses)
    rec["n_confirmed"] = sum(h.cross_domain_confirmed for h in case.hypotheses)
    rec["n_det_A2"] = len(res.evidence[0].detections)
    rec["n_det_A3"] = len(res.evidence[1].detections)

    methods = {
        "A4_confirmed":   _case_to_dets(case, {HypothesisStatus.supported}),
        "A4_conf_or_amb": _case_to_dets(case, {HypothesisStatus.supported,
                                               HypothesisStatus.ambiguous}),
    }
    rec["scores"] = {name: _metrics(gpr_eval.evaluate(dets, targets, mode="depth"))
                     for name, dets in methods.items()}
    rec["case_note"] = case.overall_note
    rec["hyps"] = [{"id": h.id, "status": h.status.value, "conf": h.confidence,
                    "x_m": h.x_m, "depth_m": h.depth_m,
                    "confirmed": h.cross_domain_confirmed,
                    "statement": h.statement} for h in case.hypotheses]
    rec["gt_depths"] = sorted({round(float(t["depth_m"]), 2) for t in targets})
    return rec


def _checkpoint(done, failed, meta):
    VLM_JSON.write_text(json.dumps({**meta, "failed": failed, "lines": done},
                                   indent=2), encoding="utf-8")


def main(db_root: str | None = None) -> dict:
    from gpr_bench import eval as gpr_eval
    from gpr_bench.datasets.tu1208 import TU1208Adapter
    from gpr_bench.datasets.tu1208.filename import parse_name

    fresh = "--fresh" in sys.argv
    ad = TU1208Adapter(root=db_root) if db_root else TU1208Adapter()

    # SAME scorable set as the oracle sweep: dt_ns present + has GT. Reuse the oracle
    # JSON's file list so the two reports cover identical lines.
    oracle = json.loads(ORACLE_JSON.read_text(encoding="utf-8")) if ORACLE_JSON.exists() else {"lines": []}
    ora_by = {r["file"]: r for r in oracle.get("lines", [])}
    scorable_files = [r["file"] for r in oracle.get("lines", []) if r["dt_ns"] and r["n_gt"] > 0]

    # map basename -> full path from the adapter (dedup by basename, first wins)
    path_by = {}
    for ln in ad.lines():
        path_by.setdefault(Path(ln).name.lower(), ln)
    targets_list = [(f, path_by[f.lower()]) for f in scorable_files if f.lower() in path_by]

    done, failed = [], []
    if not fresh and VLM_JSON.exists():
        prev = json.loads(VLM_JSON.read_text(encoding="utf-8"))
        done = prev.get("lines", []); failed = prev.get("failed", [])
        have = {r["file"] for r in done}
        targets_list = [(f, p) for f, p in targets_list if f not in have]
        print(f"[resume] {len(done)} lines already done; {len(targets_list)} remaining")

    meta = {"model": MODEL, "root": str(ad.root), "n_scorable": len(scorable_files)}
    print(f"[VLM agentic sweep] model={MODEL} | {len(scorable_files)} scorable lines "
          f"| {len(targets_list)} to run\n")

    for i, (fname, line) in enumerate(targets_list, 1):
        try:
            rec = run_one(ad, line, gpr_eval, parse_name)
            done.append(rec)
            sc = rec["scores"]["A4_conf_or_amb"]
            osc = (ora_by.get(fname, {}).get("scores", {}).get("A4_conf_or_amb", {}))
            tok = rec["usage"]
            print(f"  [{i:2d}/{len(targets_list)}] {fname:34s} {rec['region'] or '?':11s} "
                  f"VLM P={sc['precision']:.2f} R={sc['recall']:.2f} AP={sc['ap']:.2f} "
                  f"| oracle R={osc.get('recall', float('nan')):.2f} "
                  f"| {rec['secs']:.0f}s tok={tok['input']}/{tok['output']}")
        except Exception as e:  # noqa: BLE001 -- API failures must not lose the sweep
            failed.append({"file": fname, "error": repr(e)[:300]})
            print(f"  [{i:2d}/{len(targets_list)}] {fname:34s} FAILED: {e!r}")
            traceback.print_exc()
        _checkpoint(done, failed, meta)   # persist after every line

    # ---- aggregate over completed VLM lines (scorable by construction) ----
    rows = [r for r in done if r.get("scores")]
    methods = ["A4_confirmed", "A4_conf_or_amb"]
    agg = {m: _aggregate(rows, m) for m in methods} if rows else {}
    ora_rows = [ora_by[r["file"]] for r in rows if r["file"] in ora_by]
    ora_agg = {m: _aggregate(ora_rows, m) for m in methods} if ora_rows else {}

    print("\n" + "=" * 78)
    print(f"VLM lines done: {len(rows)} | failed: {len(failed)}")
    tin = sum(r["usage"]["input"] for r in rows)
    tout = sum(r["usage"]["output"] for r in rows)
    treq = sum(r["usage"]["requests"] for r in rows)
    print(f"total tokens: in={tin:,} out={tout:,} | requests={treq} | "
          f"wall={sum(r['secs'] for r in rows):.0f}s")

    print("\n--- OVERALL (micro over completed lines) : VLM vs ORACLE ---")
    hdr = f"{'method':16s} {'engine':7s} {'tp':>4} {'fp':>4} {'fn':>4} {'mP':>5} {'mR':>5} {'mF1':>5} {'mAP':>5} {'dMAE':>6}"
    print(hdr); print("-" * len(hdr))
    for m in methods:
        for eng, a in [("VLM", agg.get(m)), ("oracle", ora_agg.get(m))]:
            if not a:
                continue
            mae = f"{a['depth_mae']:.3f}" if a["depth_mae"] is not None else "-"
            print(f"{m:16s} {eng:7s} {a['tp']:4d} {a['fp']:4d} {a['fn']:4d} "
                  f"{a['micro_precision']:5.2f} {a['micro_recall']:5.2f} {a['micro_f1']:5.2f} "
                  f"{a['mean_AP']:5.2f} {mae:>6}")

    print("\n--- PER-REGION (A4 confirmed+ambiguous, VLM micro) ---")
    regions = sorted({r["region"] for r in rows if r["region"]})
    per_region = {}
    rhdr = f"{'region':14s} {'#ln':>3} {'tp':>4} {'fp':>4} {'fn':>4} {'mP':>5} {'mR':>5} {'mAP':>5} {'dMAE':>6}"
    print(rhdr); print("-" * len(rhdr))
    for reg in regions:
        sub = [r for r in rows if r["region"] == reg]
        a = _aggregate(sub, "A4_conf_or_amb")
        per_region[reg] = a
        mae = f"{a['depth_mae']:.3f}" if a["depth_mae"] is not None else "-"
        print(f"{reg:14s} {a['n_lines']:3d} {a['tp']:4d} {a['fp']:4d} {a['fn']:4d} "
              f"{a['micro_precision']:5.2f} {a['micro_recall']:5.2f} {a['mean_AP']:5.2f} {mae:>6}")

    meta.update({"overall_vlm": agg, "overall_oracle_same_lines": ora_agg,
                 "per_region_vlm": per_region,
                 "tokens": {"input": tin, "output": tout, "requests": treq},
                 "n_done": len(rows)})
    _checkpoint(done, failed, meta)
    print(f"\nWrote -> {VLM_JSON}")
    return meta


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(pos[0] if pos else None)

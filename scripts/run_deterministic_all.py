"""Run the FROZEN deterministic oracle on EVERY TU1208 line and report.

`run_deterministic_loop.py` runs the keyless A1->(A2||A3)->A4 loop on a single line.
This driver points the TU1208 adapter at the full IFSTTAR `Database_2018` (when
present on disk) and runs that SAME frozen oracle over every loadable radargram,
scoring each against the per-region ground truth (depth match, dz_tol=0.15 m), then
aggregating per-region and overall. Nothing is tuned -- this is a report, not a fit
(see deterministic.py FROZEN banner + no-calibration first principle).

Lines are de-duplicated by basename (the Zenodo package copies the LIMESTONE files
into other region folders). Lines whose loader reports no `dt_ns` have no recoverable
depth axis -> flagged `no_timebase` and EXCLUDED from the scored aggregates (kept in
the raw dump). MULTI-LAYER carries 0 point targets (a false-alarm control) -> excluded
from recall/AP aggregates, reported separately.

Run:
    PYTHONPATH="<GPR-Bench>;<GPR-Tools>/src" python run_deterministic_all.py [db_root]
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import run_agent_validation as base
from gpr_agent.contracts import HypothesisStatus
from gpr_agent.deterministic import orchestrate_deterministic
from gpr_agent.session import Session

ART = Path(__file__).resolve().parent / "_artifacts"
ART.mkdir(exist_ok=True)


def _case_to_dets(case, statuses):
    from gpr_bench.eval import Detection
    return [Detection(x_m=h.x_m, depth_m=h.depth_m, conf=h.confidence)
            for h in case.hypotheses if h.status in statuses and h.depth_m is not None]


def _metrics(report):
    mae = (report.get("localization") or {}).get("depth_mae")
    return {"n_det": report["count_pred"], "tp": report["tp"], "fp": report["fp"],
            "fn": report["fn"], "precision": report["precision"],
            "recall": report["recall"], "f1": report["f1"],
            "ap": report["average_precision"], "depth_mae": mae}


def run_one(ad, line, gpr_eval, parse_name):
    fm = parse_name(Path(line).name)
    rg = ad.load(line)
    targets = [t for t in ad.targets(line) if t.get("depth_m") is not None]
    rec = {"file": Path(line).name, "region": fm.region, "line": fm.line,
           "freq_mhz": fm.freq_mhz, "vendor": fm.vendor,
           "n_samples": int(rg.data.shape[0]), "n_traces": int(rg.data.shape[1]),
           "dt_ns": rg.dt_ns, "dx_m": rg.dx_m, "n_gt": len(targets)}

    s = Session(arrays={"raw": rg.data})
    res = orchestrate_deterministic(s, "raw", dt_ns=rg.dt_ns or 1.0,
                                    dx_m=rg.dx_m or 1.0, max_rounds=2)
    case = res.case
    rec["rounds"] = res.rounds
    rec["n_hyp"] = len(case.hypotheses)
    rec["n_confirmed"] = sum(h.cross_domain_confirmed for h in case.hypotheses)

    methods = {
        "A2_apex":        base.perceive_a2_radargram(rg),
        "A4_confirmed":   _case_to_dets(case, {HypothesisStatus.supported}),
        "A4_conf_or_amb": _case_to_dets(case, {HypothesisStatus.supported,
                                               HypothesisStatus.ambiguous}),
    }
    rec["scores"] = {name: _metrics(gpr_eval.evaluate(dets, targets, mode="depth"))
                     for name, dets in methods.items()}
    # keep a compact view of the A4 case for qualitative reporting
    rec["case_note"] = case.overall_note
    rec["hyps"] = [{"id": h.id, "status": h.status.value, "conf": h.confidence,
                    "x_m": h.x_m, "depth_m": h.depth_m,
                    "confirmed": h.cross_domain_confirmed,
                    "statement": h.statement} for h in case.hypotheses]
    rec["gt_depths"] = sorted({round(float(t["depth_m"]), 2) for t in targets})
    return rec


def _aggregate(rows, method):
    """Micro aggregate (sum tp/fp/fn across lines) for one method over `rows`."""
    tp = sum(r["scores"][method]["tp"] for r in rows)
    fp = sum(r["scores"][method]["fp"] for r in rows)
    fn = sum(r["scores"][method]["fn"] for r in rows)
    aps = [r["scores"][method]["ap"] for r in rows]
    maes = [r["scores"][method]["depth_mae"] for r in rows
            if r["scores"][method]["depth_mae"] is not None]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n_lines": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": prec, "micro_recall": rec, "micro_f1": f1,
            "mean_AP": (sum(aps) / len(aps) if aps else 0.0),
            "depth_mae": (sum(maes) / len(maes) if maes else None)}


def main(db_root: str | None = None) -> dict:
    from gpr_bench import eval as gpr_eval
    from gpr_bench.datasets.tu1208 import TU1208Adapter
    from gpr_bench.datasets.tu1208.filename import parse_name

    ad = TU1208Adapter(root=db_root) if db_root else TU1208Adapter()
    all_lines = ad.lines()
    # de-dup by lowercased basename (Zenodo copies LIMESTONE files into other folders)
    seen, lines = set(), []
    for ln in all_lines:
        key = Path(ln).name.lower()
        if key in seen:
            continue
        seen.add(key); lines.append(ln)
    print(f"[TU1208] root={ad.root}\n  {len(all_lines)} data files -> {len(lines)} unique lines\n")

    rows, failed = [], []
    for i, ln in enumerate(lines, 1):
        name = Path(ln).name
        try:
            rec = run_one(ad, ln, gpr_eval, parse_name)
            rows.append(rec)
            sc = rec["scores"]["A4_conf_or_amb"]
            tb = "" if rec["dt_ns"] else "  [NO TIMEBASE]"
            print(f"  [{i:2d}/{len(lines)}] {name:36s} {rec['region'] or '?':12s} "
                  f"gt={rec['n_gt']:2d} A4det={sc['n_det']:2d} "
                  f"P={sc['precision']:.2f} R={sc['recall']:.2f} AP={sc['ap']:.2f}{tb}")
        except Exception as e:  # noqa: BLE001 -- one bad line must not kill the sweep
            failed.append({"file": name, "error": repr(e)})
            print(f"  [{i:2d}/{len(lines)}] {name:36s} FAILED: {e!r}")
            traceback.print_exc()

    # ---- partition rows for honest aggregation ----
    scorable = [r for r in rows if r["dt_ns"] and r["n_gt"] > 0]   # usable depth axis + GT
    no_tb = [r for r in rows if not r["dt_ns"]]
    no_gt = [r for r in rows if r["dt_ns"] and r["n_gt"] == 0]     # MULTI-LAYER control

    print("\n" + "=" * 78)
    print(f"SCORABLE lines (have dt_ns + GT): {len(scorable)} | "
          f"no-timebase: {len(no_tb)} | no-GT control: {len(no_gt)} | failed: {len(failed)}")

    methods = ["A2_apex", "A4_confirmed", "A4_conf_or_amb"]
    agg = {m: _aggregate(scorable, m) for m in scorable[0]["scores"]} if scorable else {}
    print("\n--- OVERALL (micro-averaged over scorable lines) ---")
    hdr = f"{'method':16s} {'tp':>4} {'fp':>4} {'fn':>4} {'mP':>5} {'mR':>5} {'mF1':>5} {'mAP':>5} {'dMAE':>6}"
    print(hdr); print("-" * len(hdr))
    for m in methods:
        a = agg[m]
        mae = f"{a['depth_mae']:.3f}" if a["depth_mae"] is not None else "-"
        print(f"{m:16s} {a['tp']:4d} {a['fp']:4d} {a['fn']:4d} {a['micro_precision']:5.2f} "
              f"{a['micro_recall']:5.2f} {a['micro_f1']:5.2f} {a['mean_AP']:5.2f} {mae:>6}")

    # ---- per-region (A4_conf_or_amb) ----
    regions = sorted({r["region"] for r in scorable if r["region"]})
    per_region = {}
    print("\n--- PER-REGION (A4 confirmed+ambiguous, micro) ---")
    rhdr = f"{'region':14s} {'#ln':>3} {'tp':>4} {'fp':>4} {'fn':>4} {'mP':>5} {'mR':>5} {'mAP':>5} {'dMAE':>6}"
    print(rhdr); print("-" * len(rhdr))
    for reg in regions:
        sub = [r for r in scorable if r["region"] == reg]
        a = _aggregate(sub, "A4_conf_or_amb")
        per_region[reg] = a
        mae = f"{a['depth_mae']:.3f}" if a["depth_mae"] is not None else "-"
        print(f"{reg:14s} {a['n_lines']:3d} {a['tp']:4d} {a['fp']:4d} {a['fn']:4d} "
              f"{a['micro_precision']:5.2f} {a['micro_recall']:5.2f} {a['mean_AP']:5.2f} {mae:>6}")

    # no-GT control: how many spurious detections does the oracle emit where there is nothing?
    if no_gt:
        det_ctrl = sum(r["scores"]["A4_conf_or_amb"]["n_det"] for r in no_gt)
        print(f"\nNo-GT control (MULTI-LAYER, {len(no_gt)} lines): "
              f"{det_ctrl} A4 detections emitted where 0 point targets exist "
              f"(the detector always proposes; these are all false alarms by construction).")

    out = {"root": str(ad.root), "n_unique_lines": len(lines),
           "counts": {"scorable": len(scorable), "no_timebase": len(no_tb),
                      "no_gt_control": len(no_gt), "failed": len(failed)},
           "overall": agg, "per_region": per_region,
           "no_timebase_files": [r["file"] for r in no_tb],
           "failed": failed, "lines": rows}
    (ART / "tu1208_all_deterministic.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote per-line + aggregate JSON -> {ART / 'tu1208_all_deterministic.json'}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

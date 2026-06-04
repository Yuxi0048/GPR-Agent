"""Sanity check: can the keyless deterministic workflow analyze the generated corpus?

For a sample of generated scenes, runs orchestrate_deterministic (A1 image -> A2/A3
detect -> A4 fuse) on bscan.npy and compares the hypotheses to the GT labels. NOT a
benchmark and nothing is tuned -- it answers "do the B-scans contain analyzable targets,
and does the workflow run on them?" Reports the scan-coverage caveat explicitly (the
scan window vs object placement).

PYTHONPATH = GPR-Agent/src;GPR-Sim/src;GPR-KnowledgeBase;GPR-Tools/src
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from gpr_agent.deterministic import orchestrate_deterministic
from gpr_agent.session import Session

CORPUS = Path("e:/github/GPR-Sim/data/generated_corpus")
SCAN_START_M = 0.10          # run_bscan default
X_TOL_M = 0.08               # x-match tolerance (scene metres)
PER_TYPE = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def _circle_gt(labels):
    """GT objects that produce a hyperbola (circles in cross-section)."""
    out = []
    for o in labels["objects"]:
        if o.get("center_x_m") is not None and o["kind"] not in ("pipe_void",):
            out.append(o)
    return out


def main():
    dirs_by_type: dict[str, list[Path]] = {}
    for d in sorted(CORPUS.glob("*/*")):
        if (d / "bscan.npy").exists() and (d / "labels.json").exists():
            dirs_by_type.setdefault(d.parent.name, []).append(d)

    print(f"{'scene_type':20s} {'host':14s} {'nT':>3s} {'scanX':>11s} "
          f"{'GT':>2s} {'in':>2s} {'det':>3s} {'xhit':>4s}  note")
    print("-" * 96)
    agg = {}
    for stype, dirs in sorted(dirs_by_type.items()):
        for d in dirs[-PER_TYPE:]:                                  # newest few per type
            labels = json.loads((d / "labels.json").read_text())
            bscan = np.load(d / "bscan.npy")                       # (n_samples, n_traces)
            n_t = bscan.shape[1]
            acq = labels["acquisition"]
            dx_m = acq["dx_m"]
            dt_ns = acq["dt_ns"]
            scan_start = acq.get("scan_start_m", SCAN_START_M)     # fixed data records it
            scan_max = scan_start + (n_t - 1) * dx_m
            gts = _circle_gt(labels)
            in_scan = [o for o in gts if scan_start <= o["center_x_m"] <= scan_max]
            try:
                sess = Session(arrays={"raw": np.asarray(bscan, float)})
                res = orchestrate_deterministic(sess, "raw", dt_ns=dt_ns, dx_m=dx_m)
                dets = [(scan_start + h.x_m, h.depth_m) for h in res.case.hypotheses
                        if h.x_m is not None]
                note = ""
            except Exception as e:
                dets, note = [], f"ERROR {type(e).__name__}: {str(e)[:60]}"
            xhit = sum(any(abs(dx0 - o["center_x_m"]) <= X_TOL_M for dx0, _ in dets)
                       for o in in_scan)
            print(f"{stype:20s} {labels['host_material']:14s} {n_t:3d} "
                  f"[{SCAN_START_M:.2f},{scan_max:.2f}] {len(gts):2d} {len(in_scan):2d} "
                  f"{len(dets):3d} {xhit:4d}  {note}")
            a = agg.setdefault(stype, [0, 0, 0, 0, 0])
            a[0] += 1; a[1] += len(gts); a[2] += len(in_scan); a[3] += len(dets); a[4] += xhit

    print("\n=== per-type totals (samples | GT | in-scan | detections | x-hits) ===")
    tot = [0, 0, 0, 0, 0]
    for stype, a in sorted(agg.items()):
        print(f"  {stype:20s} n={a[0]:2d}  GT={a[1]:3d}  in-scan={a[2]:3d}  det={a[3]:3d}  xhit={a[4]:3d}")
        for i in range(5):
            tot[i] += a[i]
    print(f"  {'TOTAL':20s} n={tot[0]:2d}  GT={tot[1]:3d}  in-scan={tot[2]:3d}  det={tot[3]:3d}  xhit={tot[4]:3d}")
    if tot[2]:
        print(f"\n  in-scan recall (x): {tot[4]}/{tot[2]} = {tot[4]/tot[2]:.0%}  "
              f"| GT off-scan: {tot[1]-tot[2]}/{tot[1]} = {(tot[1]-tot[2])/max(tot[1],1):.0%}")


if __name__ == "__main__":
    main()

"""Diagnostic: A3 (subsurface blob-NMS) is the SAME detector either way -- it's A1's
IMAGE that matters. On the un-migrated envelope A3 is flooded by a bright shallow
band and misses the deep pipes; on A1's real migrated section A3 hits them. Saves a
PNG, prints a before/after summary. (Migration is A1's job; A3 is unchanged.)

The shallow band (direct wave / ringing / a flat layer) is the BRIGHTEST energy, so
top-N-by-energy detection lands entirely on it. A horizontally-continuous row is
bright across MANY traces; a point apex is bright at ~one trace. `frac_bright[row] =
mean(env[row] > 0.5*row_max)` measures that lateral continuity (band -> large) and
explains the flood. The fix is in A1: BGR + Kirchhoff migration focuses apex energy
into blobs at the pipe depth, which is the image A3 is meant to read.

Run:
    PYTHONPATH=<GPR-Bench>;<GPR-Tools>/src  python run_a3_band_diagnostic.py [dataset]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gpr_agent import tools
from gpr_agent.session import Session

FRAC_SUPPRESS = 0.30     # row bright across >30% of the line => laterally-continuous band
GT_TOL_M = 0.15          # a detection "hits" a GT depth within this tolerance
MAX_DETS = 60


def _frac_bright(env: np.ndarray) -> np.ndarray:
    """Per-depth lateral continuity: fraction of traces brighter than half the row max."""
    row_max = env.max(axis=1, keepdims=True)
    return (env > 0.5 * row_max).mean(axis=1)


def _near_gt(depths, gt_depths):
    return sum(any(abs(z - g) <= GT_TOL_M for g in gt_depths) for z in depths)


def main(dataset: str = "tu1208") -> dict:
    global DT, DX
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gpr_bench.datasets.registry import get_adapter

    ad = get_adapter(dataset)
    line = ad.lines()[0]
    rg = ad.load(line)
    v, DT, DX = tools.NOMINAL_V, (rg.dt_ns or 1.0), (rg.dx_m or 1.0)
    gt_depths = sorted(t["depth_m"] for t in ad.targets(line) if t.get("depth_m") is not None)

    raw = np.asarray(rg.data, float)
    n_s, n_t = raw.shape
    depth = 0.5 * v * np.arange(n_s) * DT
    x_max = n_t * DX

    # A1, two ways: (a) un-migrated envelope (no BGR), (b) A1's real BGR + migrated section
    s = Session(arrays={"raw": raw})
    img_raw = tools.image(s, "raw", dt_ns=DT, dx_m=DX, velocity=v, migrate=False, bgr=False)
    env_raw = np.asarray(s.arrays["bscan"], float)
    det_raw = tools.analyze_subsurface(s, img_raw.subsurface, velocity=v, max_dets=MAX_DETS).detections

    img_mig = tools.image(s, "raw", dt_ns=DT, dx_m=DX, velocity=v, migrate="adjoint", bgr=True)
    mig = np.asarray(s.arrays["subsurface"], float)             # (nz, n_traces') focused section
    det_mig = tools.analyze_subsurface(s, img_mig.subsurface, velocity=v, max_dets=MAX_DETS).detections
    frac = _frac_bright(env_raw)

    def split(dets):
        keep, drop = [], []
        for d in dets:
            r = min(n_s - 1, int(round(d.depth_m / (0.5 * v * DT))))
            (drop if frac[r] > FRAC_SUPPRESS else keep).append((d.x_m, d.depth_m))
        return keep, drop

    keep_raw, drop_raw = split(det_raw)
    band_depth = float(np.median([z for _, z in drop_raw])) if drop_raw else float("nan")

    # ---------- figure: same A3 on un-migrated envelope (floods) vs A1's migrated section ----------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10),
                             gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
    (axA, axCoh), (axB, axHist) = axes

    def show(ax, env, title):
        ax.imshow(env, aspect="auto", cmap="gray_r", vmin=0, vmax=float(np.percentile(env, 99)),
                  extent=[0, x_max, depth[-1], depth[0]])
        for g in gt_depths:
            ax.axhline(g, color="gold", ls="--", lw=0.8, alpha=0.7)
        ax.set_xlabel("x (m)"); ax.set_ylabel("depth (m, uncal. v=0.1)"); ax.set_title(title)

    show(axA, env_raw, f"{ad.name}: A1=un-migrated envelope -- A3 floods the band (gold=GT depths)")
    if keep_raw:
        axA.scatter([x for x, _ in keep_raw], [z for _, z in keep_raw], marker="x", s=55,
                    c="limegreen", lw=1.6, label=f"point-like (frac≤{FRAC_SUPPRESS})")
    axA.scatter([x for x, _ in drop_raw], [z for _, z in drop_raw], marker="x", s=55,
                c="red", lw=1.6, label=f"band (frac>{FRAC_SUPPRESS})")
    axA.legend(loc="lower right", fontsize=8, framealpha=0.9)

    show(axB, mig, "A1=BGR + real Kirchhoff migration -- same A3, now hits the deep pipes")
    axB.scatter([d.x_m for d in det_mig], [d.depth_m for d in det_mig], marker="x", s=55,
                c="dodgerblue", lw=1.6, label="A3 blobs (migrated)")
    axB.legend(loc="lower right", fontsize=8, framealpha=0.9)

    axCoh.plot(frac, depth, color="purple", lw=1.0); axCoh.axvline(FRAC_SUPPRESS, color="red", ls=":")
    axCoh.set_xlabel("lateral continuity\nfrac bright"); axCoh.set_xlim(0, 1)
    axCoh.set_title("band = high lateral continuity")
    edges = np.linspace(depth[0], depth[-1], 50)
    axHist.hist([d.depth_m for d in det_raw], bins=edges, orientation="horizontal",
                color="0.6", label="un-migrated")
    axHist.hist([d.depth_m for d in det_mig], bins=edges, orientation="horizontal",
                color="dodgerblue", alpha=0.6, label="migrated")
    for g in gt_depths:
        axHist.axhline(g, color="gold", ls="--", lw=0.6, alpha=0.6)
    axHist.set_xlabel("A3 blob count"); axHist.set_title("blob depths vs GT"); axHist.legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_ylim(depth[-1], depth[0])

    out = Path(__file__).parent / "_artifacts" / "a3_band_diagnostic.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)

    hit_raw = _near_gt([d.depth_m for d in det_raw], gt_depths)
    hit_mig = _near_gt([d.depth_m for d in det_mig], gt_depths)
    print(f"[{ad.name}] GT depths: {[round(g,2) for g in gt_depths]}")
    print(f"UN-MIGRATED  : {len(det_raw)} A3 blobs | {len(drop_raw)} in band (~{band_depth:.2f} m) | "
          f"{hit_raw} within {GT_TOL_M} m of a GT depth")
    print(f"A1 MIGRATED  : {len(det_mig)} A3 blobs | {hit_mig} within {GT_TOL_M} m of a GT depth  <-- A1's job, A3 unchanged")
    print(f"saved: {out}")
    return {"band_depth_m": band_depth, "raw_in_band": len(drop_raw),
            "raw_hits_gt": hit_raw, "mig_hits_gt": hit_mig, "artifact": str(out)}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tu1208")

"""Validate an agent run against a GPR-Bench dataset.

The end-to-end loop the agent plugs into (see AGENTS_DESIGN.md):

    GPR-Bench dataset (radargram + ground truth)  ->  perceive(...)  ->  gpr_bench.eval

``perceive`` is the **plug point** for the agent's detectors (A2 radargram /
A3 subsurface). Until the 4-agent system is built it runs a deterministic,
**GT-independent baseline** (envelope depth-peaks) so the validation loop is
exercised *honestly* — a real, if weak, detector scored against the benchmark GT.
Swap ``perceive`` for the agent's perception when it's ready; the rest is fixed.

Run (after `make setup`, or with GPR_TOOLS_SRC set + PYTHONPATH=<GPR-Bench>):

    python run_agent_validation.py [dataset]        # default: tu1208
"""
from __future__ import annotations

import sys


def perceive_baseline(rg, *, v_m_per_ns: float = 0.1, max_dets: int = 20):
    """Deterministic baseline 'perception' (NOT the agent).

    Stacks envelope energy across traces and picks the strongest depth peaks --
    a GT-independent stand-in for the agent's A2/A3 detectors. Depth uses a
    nominal velocity (UNcalibrated -- this is a placeholder, not an estimate).
    """
    import numpy as np
    from scipy.signal import find_peaks

    from gpr_data_processing.attributes.envelope import envelope
    from gpr_bench.eval import Detection

    data = np.asarray(rg.data, float)                 # (n_samples, n_traces)
    env = np.abs(envelope(data, axis=0))              # envelope along time/sample
    profile = env.mean(axis=1)                        # depth-energy profile
    peaks, _ = find_peaks(profile, distance=max(3, profile.size // 50))
    if peaks.size == 0:
        return []
    order = peaks[np.argsort(profile[peaks])[::-1][:max_dets]]   # top-N by energy
    dt_ns = rg.dt_ns or 1.0
    pmax = float(profile.max()) or 1.0
    return [Detection(x_m=None,
                      depth_m=0.5 * v_m_per_ns * (int(s) * dt_ns),
                      conf=float(profile[int(s)] / pmax))
            for s in order]


def perceive_a2_radargram(rg, *, v_m_per_ns: float = 0.1, max_dets: int = 20,
                          nms: int = 9, thresh_pct: float = 90.0, min_depth_m: float = 0.2):
    """Deterministic A2 (radargram domain): hyperbola-apex-like detection.

    Detects the depth of each strong hyperbola **apex** via the *max-across-traces*
    envelope profile -- apex energy (one bright trace at the apex depth) survives a
    MAX projection, whereas the baseline's MEAN favours flat layers and washes
    point apices out. For each depth peak, the apex x is the brightest trace there.
    numpy/scipy only; same nominal (uncalibrated) velocity as the baseline, so only
    the detection method differs. Direct-wave/surface band skipped. Deterministic
    stand-in for the agent's A2; the VLM/tool-use A2 plugs in at the same contract.
    """
    import numpy as np
    from scipy.signal import find_peaks

    from gpr_data_processing.attributes.envelope import envelope
    from gpr_bench.eval import Detection

    data = np.asarray(rg.data, float)                 # (n_samples, n_traces)
    env = np.abs(envelope(data, axis=0))
    profile = env.max(axis=1)                         # strongest apex energy per depth
    peaks, _ = find_peaks(profile, distance=max(3, profile.size // 60))
    dt_ns = rg.dt_ns or 1.0
    dx_m = rg.dx_m or 1.0
    cand = [s for s in peaks if 0.5 * v_m_per_ns * (s * dt_ns) >= min_depth_m]
    cand.sort(key=lambda s: profile[s], reverse=True)
    cand = cand[:max_dets]
    if not cand:
        return []
    vmax = float(profile[cand].max()) or 1.0
    return [Detection(x_m=float(int(np.argmax(env[s, :])) * dx_m),
                      depth_m=float(0.5 * v_m_per_ns * (s * dt_ns)),
                      conf=float(profile[s] / vmax))
            for s in cand]


def main(dataset: str = "tu1208") -> dict:
    from gpr_bench import eval as gpr_eval
    from gpr_bench.datasets.registry import get_adapter

    ad = get_adapter(dataset)
    line = ad.lines()[0]
    rg = ad.load(line)
    targets = [t for t in ad.targets(line) if t.get("depth_m") is not None]
    print(f"[{ad.name}] line {rg.data.shape} (n_samp x n_tr); {len(targets)} GT targets with depth")

    # ---- perception methods (PLUG POINT: the agent's A2/A3 slot in here) ----
    methods = {
        "baseline (envelope stack)": perceive_baseline(rg),
        "A2 radargram (apex depths)": perceive_a2_radargram(rg),
    }

    # ---- validate each against the GPR-Bench GT via the eval core ----
    hdr = f"{'method':32s} {'n_det':>5} {'prec':>6} {'recall':>6} {'AP':>6} {'depthMAE':>9}"
    print(hdr); print("-" * len(hdr))
    reports = {}
    for name, dets in methods.items():
        r = gpr_eval.evaluate(dets, targets, mode="depth")
        reports[name] = r
        mae = r.get("localization", {}).get("depth_mae")
        mae_s = f"{mae:.3f}" if mae is not None else "-"
        print(f"{name:32s} {len(dets):5d} {r['precision']:6.2f} {r['recall']:6.2f} "
              f"{r['average_precision']:6.2f} {mae_s:>9}")
    print("\nLoop: GPR-Bench dataset -> perceive -> gpr_bench.eval (depth match, "
          "dz_tol=0.15 m). The deterministic A2 is the agent-perception stand-in; "
          "the VLM/tool-use A2/A3 plug in at the same Detection contract.")
    return reports


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tu1208")

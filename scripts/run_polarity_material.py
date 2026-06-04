"""Validate polarity -> material on the build123d/gprMax FDTD dev scenes.

Closes the loop: A2-style apex polarity (GPR-Tools `estimate_apex_polarity`) -> the
KB canonical lookup (`gpr_kb.reference.candidates_by_polarity`) -> candidate buried
materials, checked against the KNOWN simulated material. The apex is located from the
scene geometry (truth) so this isolates the POLARITY->material mapping from detection.

Physics expectation in soil (eps_r~10): metal & air-void read NEGATIVE (reversed);
water-filled reads POSITIVE. Polarity NARROWS the candidate set, it does not uniquely
ID the material -- success = the true material's class is IN the candidate set.

Run:
    PYTHONPATH=<GPR-Tools>/src;<GPR-KnowledgeBase>  python run_polarity_material.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path("e:/github/_gprtools_tmp")
_C0 = 0.3  # m/ns
# the true material -> its object-taxonomy class + the polarity we expect to read
# scene dir -> (object-taxonomy class, expected polarity, scene fc_hz). Host soil is
# read from each scene.json, so concrete shows HOST-DEPENDENCE: reversed in moist
# limestone (eps~10 > 7), normal in dry sand (eps~4 < 7). High-eps water runs at
# 250 MHz so gprMax samples its short wavelength without numerical dispersion.
_EXPECT = {
    "metal":            ("steel_pipe_empty", "NEGATIVE", 1.0e9),
    "air":              ("air_void", "NEGATIVE", 1.0e9),
    "water":            ("water_pocket", "POSITIVE", 2.5e8),
    "pvc":              ("pvc_pipe_empty", "NEGATIVE", 1.0e9),
    "hdpe":             ("hdpe_pipe_empty", "NEGATIVE", 1.0e9),
    "concrete_moist":   ("concrete_pipe_empty", "NEGATIVE", 1.0e9),   # eps7 < soil eps10
    "concrete_drysand": ("concrete_pipe_empty", "POSITIVE", 1.0e9),   # eps7 > sand eps4
}


def _dt_ns(scene_dir: Path) -> float:
    import h5py
    with h5py.File(scene_dir / "t000.out", "r") as f:
        return float(f.attrs["dt"]) * 1e9


def _apex_index(bscan, scene, dt_ns, fc_hz):
    """Locate the pipe apex (trace, sample) from geometry truth + a local energy max."""
    from gpr_data_processing.attributes.envelope import envelope
    env = np.abs(envelope(np.asarray(bscan, float), axis=0))
    n_s, n_t = env.shape
    v_soil = _C0 / np.sqrt(scene["soil_eps"])
    # expected apex trace + two-way time (air gap then soil to the pipe)
    ti = int(round((scene["center_x_m"] - scene["scan_start_m"]) / scene["scan_step_m"]))
    ti = max(0, min(n_t - 1, ti))
    twtt = 2 * (scene["air_gap_above_m"] / _C0 + scene["depth_m"] / v_soil)   # path time
    ricker_ns = 1.0 / (fc_hz * 1e-9)                                          # one ricker period
    # the wavelet peak lags the path time by ~1 ricker period; search path..path+3 periods
    lo = max(0, int(twtt / dt_ns) - 20)
    hi = min(n_s, int((twtt + 3.0 * ricker_ns) / dt_ns) + 10)
    si = lo + int(np.argmax(env[lo:hi, ti]))
    return ti, si


def main() -> dict:
    from gpr_data_processing.detection.material_plausibility import estimate_apex_polarity, Polarity
    from gpr_kb import reference as ref

    print(f"{'scene':6} {'true':16} {'polarity':10} {'conf':5} {'expect':8} {'class in candidates?':22}")
    print("-" * 78)
    results = {}
    for mat, (true_obj, want, fc) in _EXPECT.items():
        d = _ROOT / f"cad_{mat}"
        if not (d / "bscan.npy").exists():
            print(f"{mat:6} (not generated yet)"); continue
        bscan = np.load(d / "bscan.npy")
        sj = json.loads((d / "scene.json").read_text())
        # geometry truth for apex location (the runner used these)
        scene = {"soil_eps": ref.material_eps_r_mid(sj["soil_material"]), "center_x_m": 0.40,
                 "scan_start_m": 0.10, "scan_step_m": 0.02, "depth_m": 0.15,
                 "air_gap_above_m": (sj["soil_depth_m"] + 0.04) - sj["soil_depth_m"]}
        dt_ns = _dt_ns(d)
        ti, si = _apex_index(bscan, scene, dt_ns, fc_hz=fc)
        frame = np.ascontiguousarray(np.asarray(bscan, float).T)        # (n_traces, n_samples)
        # anchor polarity to the DIRECT WAVE (raw frame, pre-BGR) -> absolute reflection
        # sign relative to the source. The FDTD direct wave is Ricker-delayed (~100+
        # samples), so widen the search window to everything before the apex.
        dw = max(60, si - 80)
        est = estimate_apex_polarity(frame, apex_ti=ti, apex_si=si, fc_hz=fc, dt_s=dt_ns * 1e-9,
                                     reference_frame=frame, direct_wave_window=dw)
        sign = {Polarity.POSITIVE: 1, Polarity.NEGATIVE: -1, Polarity.AMBIGUOUS: 0}[est.polarity]
        cand_ids = [sid for sid, _ in ref.candidates_by_polarity(sign, scene["soil_eps"])]
        in_set = true_obj in cand_ids
        ok_pol = est.polarity.value == want
        print(f"{mat:6} {true_obj:16} {est.polarity.value:10} {est.confidence:4.2f}  {want:8} "
              f"{'YES' if in_set else 'NO':4} (polarity {'ok' if ok_pol else 'MISREAD'})")
        # MULTI-CUE DIAGNOSIS: polarity + apex amplitude signature (weak prior) -> kind/material
        from gpr_agent import materials, tools
        from gpr_agent.contracts import Hypothesis
        from gpr_agent.session import Session
        amp = tools.apex_amplitude(Session(arrays={"signed": np.asarray(bscan, float)}),
                                   "signed", si, ti, fc_hz=fc, dt_ns=dt_ns)
        h = Hypothesis(id=mat, statement="apex", depth_m=scene["depth_m"])
        materials.diagnose_hypothesis(h, est.polarity.value, scene["soil_eps"],
                                      conductor=amp["conductor_like"], amplitude=amp["strength"])
        diag = h.statement.split("DIAGNOSIS:", 1)[-1].split(";")[0].strip() if "DIAGNOSIS:" in h.statement else h.kind
        true_kind = "pipe" if "pipe" in true_obj else true_obj.split("_")[-1]
        print(f"       amp={amp['strength']}/cond={amp['conductor_like']} (snr {amp['snr_db']}dB ring {amp['ringing']})"
              f" -> {diag}  (true: {true_kind})")
        results[mat] = {"polarity": est.polarity.value, "expected": want, "polarity_ok": ok_pol,
                        "true_in_candidates": in_set, "candidates": cand_ids[:5],
                        "diagnosed_kind": h.kind, "diagnosed_material": h.material}
    npass = sum(r["polarity_ok"] and r["true_in_candidates"] for r in results.values())
    print(f"\n{npass}/{len(results)} scenes: polarity read correctly AND true material in the candidate set.")
    print("NOTE: polarity narrows the set (metal/void both NEGATIVE); amplitude+depth disambiguate further.")
    return results


if __name__ == "__main__":
    main()

"""Deterministic tool bodies for the 4-agent system (the perception backbone).

These are the SAME detectors the VLM agents call as `@agent.tool`s AND the keyless
deterministic engine (`deterministic.py`) runs directly -- one source of truth for
perception, so the no-key loop and the VLM loop can't drift. numpy/scipy +
`gpr_data_processing` (envelope, dip); heavy deps imported lazily so this module
loads without them. Everything here is GT-independent and API-key-free.

A1 (`image`) does the imaging: BGR -> envelope B-scan, and BGR -> real Kirchhoff/LSM
migration (`gpr_data_processing.migration`) -> a true subsurface section. So A2
(B-scan, time) and A3 (migrated section, depth) are *geometrically* independent --
A3 detects on focused apex energy, not a depth-relabelled stretch. `migrate=False`
falls back to the depth-relabelled envelope (cheap; for tests / no-pylops envs).
Migration is A1's job; A3 only detects on whatever section A1 produces.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .contracts import (
    Detection, DomainEvidence, ImageRef, ImagingResult, Provenance, Region, ToolCall,
)

NOMINAL_V = 0.1   # m/ns -- uncalibrated default velocity (honesty flag stays False)


def _envelope(data: np.ndarray, axis: int = 0) -> np.ndarray:
    from gpr_data_processing.attributes.envelope import envelope
    return np.abs(envelope(np.asarray(data, float), axis=axis))


def _depth_m(sample: float, dt_ns: float, velocity: float) -> float:
    """Two-way-time sample index -> depth (m). Uncalibrated unless velocity is."""
    return 0.5 * velocity * (sample * dt_ns)


# --------------------------------------------------------------------------- #
# A1 -- imaging (preprocess -> B-scan; BGR + migration -> subsurface section)
# --------------------------------------------------------------------------- #
_MIGRATE_MODES = {"adjoint": "adjoint_only", "ista": "ista", "lsm": "ista", "full": "full"}


def _migration_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("pylops") is not None


def image(session, raw_handle: str, *, dt_ns: float, dx_m: float,
          velocity: float = NOMINAL_V, fc_hz: float | None = None,
          bgr: bool = True, migrate: str | bool = "adjoint", offset_m: float = 0.1,
          nz: int | None = None, zero_time: int = 0, max_traces: int = 220) -> ImagingResult:
    """A1: preprocess (BGR) -> envelope B-scan, and MIGRATE -> a real subsurface section.

    Migration is A1's job (A3 only detects on the result). `migrate` selects the
    engine path: 'adjoint' (cheap Kirchhoff), 'ista'/'lsm' (least-squares), 'full'
    (FISTA L1), or False (fall back to the depth-relabelled envelope). Traces are
    decimated to <= `max_traces` for a tractable Kirchhoff. Depth bin dz = 0.5*v*dt,
    so the subsurface depth axis stays consistent with `_depth_m`. `velocity_calibrated`
    stays False (nominal velocity -- run autofocus to earn calibration).
    """
    raw = np.asarray(session.arrays[raw_handle], float)        # (n_samples, n_traces)
    n_s, n_t = raw.shape
    pre = raw
    tool_log = [ToolCall(tool="envelope", params={"axis": 0})]
    if bgr:
        from gpr_data_processing.preprocessing.bgr import background_removal
        pre = np.asarray(background_removal(raw, window=n_t), float)   # kill flat clutter/direct wave
        tool_log.insert(0, ToolCall(tool="background_removal", params={"window": n_t}))

    bscan = _envelope(pre, axis=0)
    session.arrays["bscan"] = bscan
    bscan_ref = ImageRef(domain="radargram", handle="bscan", n_samples=n_s, n_traces=n_t, dt_ns=dt_ns, dx_m=dx_m)

    mode = _MIGRATE_MODES.get(migrate) if migrate else None
    if mode and _migration_available():
        from gpr_data_processing.migration.acquisition import AcquisitionGeometry, build_migrate_fn

        dec = max(1, int(np.ceil(n_t / max_traces)))           # decimate traces for a tractable op
        frame = np.ascontiguousarray(pre.T[::dec])             # (n_traces', n_samples) the engine wants
        dx_eff = dx_m * dec
        geo = AcquisitionGeometry.from_frame(frame=frame, dt_s=dt_ns * 1e-9, dx_m=dx_eff,
                                             fc_hz=fc_hz, offset_m=offset_m)
        out_nz = int(nz or n_s)
        migrate_fn = build_migrate_fn(geo, nz=out_nz, zero_time=int(zero_time), mode=mode)
        migrated = np.asarray(migrate_fn(frame, velocity * 1e9))           # (n_traces', nz), m/s velocity
        sub = _envelope(np.ascontiguousarray(migrated.T), axis=0)          # (nz, n_traces') focused energy
        session.arrays["subsurface"] = sub
        sub_ref = ImageRef(domain="subsurface", handle="subsurface", n_samples=out_nz,
                           n_traces=frame.shape[0], dt_ns=dt_ns, dx_m=dx_eff)
        tool_log.append(ToolCall(tool="migrate_pylops", params={"mode": mode, "decimate": dec, "nz": out_nz}))
        note = f"BGR={bgr}; real {mode} migration (Kirchhoff), trace-decimate {dec}; nominal (uncalibrated) velocity"
    else:                                                       # fallback: depth-relabelled envelope
        session.arrays["subsurface"] = bscan
        sub_ref = ImageRef(domain="subsurface", handle="subsurface", n_samples=n_s, n_traces=n_t, dt_ns=dt_ns, dx_m=dx_m)
        note = (f"BGR={bgr}; subsurface = depth-relabelled envelope "
                f"({'migration off' if not mode else 'pylops unavailable'}); nominal velocity")

    if hasattr(session, "images"):                             # let VLM tools resolve dt_ns/dx_m by handle
        session.images[bscan_ref.handle] = bscan_ref
        session.images[sub_ref.handle] = sub_ref
    return ImagingResult(bscan=bscan_ref, subsurface=sub_ref, velocity_m_per_ns=velocity,
                         velocity_calibrated=False, provenance=Provenance(agent="A1", tools=tool_log, notes=note))


# --------------------------------------------------------------------------- #
# A2 -- radargram domain (1-D apex-profile peaks in time)
# --------------------------------------------------------------------------- #
def analyze_radargram(session, img: ImageRef, *, velocity: float, max_dets: int = 20,
                      min_depth_m: float = 0.2) -> DomainEvidence:
    """A2: detect hyperbola apices via the max-across-traces envelope profile.

    Apex energy (one bright trace at the apex depth) survives a MAX projection; the
    apex x is the brightest trace at that depth. Native axis is twtt; depth is the
    uncalibrated conversion. Every detection cites its tools (evidence-based).
    """
    from scipy.signal import find_peaks

    env = np.asarray(session.arrays[img.handle], float)
    dt_ns, dx_m = (img.dt_ns or 1.0), (img.dx_m or 1.0)
    profile = env.max(axis=1)
    peaks, _ = find_peaks(profile, distance=max(3, profile.size // 60))
    cand = [int(s) for s in peaks if _depth_m(s, dt_ns, velocity) >= min_depth_m]
    cand.sort(key=lambda s: profile[s], reverse=True)
    cand = cand[:max_dets]
    vmax = float(profile[cand].max()) if cand else 1.0
    dets: list[Detection] = []
    near_floor = []
    for i, s in enumerate(cand):
        boundary = s >= env.shape[0] - 2
        if boundary:
            near_floor.append(i)
        dets.append(Detection(
            id=f"r{i:03d}", domain="radargram",
            x_m=float(int(np.argmax(env[s, :])) * dx_m),
            twtt_ns=float(s * dt_ns), depth_m=round(_depth_m(s, dt_ns, velocity), 3),
            kind="apex",
            confidence=round(float(profile[s] / vmax) * 0.95, 3),   # relative rank, <1.0
            supporting_tools=["envelope", "find_peaks"],
        ))
    quality = f"{len(dets)} apices; max-trace envelope profile"
    if near_floor:
        quality += f"; {len(near_floor)} near time-window floor (possible boundary artefact)"
    return DomainEvidence(domain="radargram", detections=dets, quality=quality,
                          provenance=Provenance(agent="A2",
                                                tools=[ToolCall(tool="envelope"), ToolCall(tool="find_peaks")]))


# --------------------------------------------------------------------------- #
# A3 -- subsurface domain (2-D blob NMS in depth)
# --------------------------------------------------------------------------- #
def analyze_subsurface(session, img: ImageRef, *, velocity: float, max_dets: int = 20,
                       min_depth_m: float = 0.2, nbhd: int = 9, thresh_pct: float = 92.0,
                       region: Optional[Region] = None) -> DomainEvidence:
    """A3: detect collapsed point reflectors via 2-D local-maxima (NMS) on the image.

    A genuinely different algorithm from A2 (2-D neighbourhood max + percentile gate),
    so agreement with A2 is informative. `region` (x0,z0,x1,z1 in m) + a lower
    `thresh_pct` let A4's bounded re-plan re-probe a specific area more sensitively.
    """
    from scipy.ndimage import maximum_filter

    env = np.asarray(session.arrays[img.handle], float)
    dt_ns, dx_m = (img.dt_ns or 1.0), (img.dx_m or 1.0)
    mx = maximum_filter(env, size=nbhd)
    thr = float(np.percentile(env, thresh_pct))
    ys, xs = np.nonzero((env == mx) & (env >= thr))             # (sample, trace) of local maxima
    items = []
    for s, t in zip(ys.tolist(), xs.tolist()):
        depth, x = _depth_m(s, dt_ns, velocity), t * dx_m
        if depth < min_depth_m:
            continue
        if region is not None:
            x0, z0, x1, z1 = region
            if not (min(x0, x1) <= x <= max(x0, x1) and min(z0, z1) <= depth <= max(z0, z1)):
                continue
        items.append((float(env[s, t]), s, x, depth))
    items.sort(reverse=True)
    items = items[:max_dets]
    vmax = items[0][0] if items else 1.0
    dets = [Detection(id=f"s{i:03d}", domain="subsurface", x_m=round(x, 3),
                      depth_m=round(depth, 3), kind="blob",
                      confidence=round(val / vmax * 0.95, 3),
                      supporting_tools=["envelope", "maximum_filter"])
            for i, (val, s, x, depth) in enumerate(items)]
    q = f"{len(dets)} blobs; 2-D NMS (nbhd={nbhd}, p{thresh_pct:g})"
    if region is not None:
        q += f"; re-probe region {tuple(round(v, 2) for v in region)}"
    return DomainEvidence(domain="subsurface", detections=dets, quality=q,
                          provenance=Provenance(agent="A3",
                                                tools=[ToolCall(tool="envelope"),
                                                       ToolCall(tool="maximum_filter",
                                                                params={"size": nbhd, "thresh_pct": thresh_pct})]))


# --------------------------------------------------------------------------- #
# Shared attribute tool (A2 may cite it; here as a small numeric summary)
# --------------------------------------------------------------------------- #
def dip_summary(session, handle: str) -> dict[str, Any]:
    """Structure-tensor dip on an image -> a small numeric summary (not arrays)."""
    from gpr_data_processing.attributes.dip import compute_dip_2d

    arr = np.asarray(session.arrays[handle], float)
    dip = np.asarray(compute_dip_2d(arr).dip_degrees, float)
    return {"dip_mean_deg": round(float(np.nanmean(dip)), 2),
            "dip_abs_p90_deg": round(float(np.nanpercentile(np.abs(dip), 90)), 2),
            "tool": "compute_dip_2d"}


def apex_polarity(session, signed_handle: str, sample_idx: int, trace_idx: int, *,
                  fc_hz: float, dt_ns: float) -> dict[str, Any]:
    """A2 hyperbola-apex polarity via GPR-Tools `estimate_apex_polarity`.

    Reads the RAW (pre-BGR, signed) frame at `signed_handle` so the direct wave is
    present as the polarity reference. Returns POSITIVE/NEGATIVE/AMBIGUOUS + confidence.
    Validated on FDTD truth in ../run_polarity_material.py (metal/void NEGATIVE, water
    POSITIVE). Feed the result + host eps_r to `materials.candidates` for material.
    """
    from gpr_data_processing.detection.material_plausibility import estimate_apex_polarity

    frame = np.ascontiguousarray(np.asarray(session.arrays[signed_handle], float).T)  # (n_traces, n_samples)
    dw = max(60, int(sample_idx) - 80)               # FDTD direct wave is Ricker-delayed
    est = estimate_apex_polarity(frame, apex_ti=int(trace_idx), apex_si=int(sample_idx),
                                 fc_hz=fc_hz, dt_s=dt_ns * 1e-9, reference_frame=frame,
                                 direct_wave_window=dw)
    return {"polarity": est.polarity.value, "confidence": round(float(est.confidence), 3),
            "snr_db": round(float(est.snr_db), 1), "tool": "estimate_apex_polarity"}


def apex_amplitude(session, signed_handle: str, sample_idx: int, trace_idx: int, *,
                   fc_hz: float, dt_ns: float) -> dict[str, Any]:
    """A2 apex amplitude SIGNATURE -- a WEAK, RELATIVE cue (never an absolute claim).

    Absolute amplitude is confounded by gain/AGC, depth, attenuation, target size, and
    coupling, so this uses only RELATIVE measures:
      - `snr_db`        : apex envelope vs the trace's median (noise) level;
      - `rel_to_direct` : apex envelope / the direct-wave envelope (scene-normalised);
      - `ringing`       : tail energy after the main lobe / main-lobe energy (a CONDUCTOR
                          resonates; this is the most discriminating but least reliable part).
    Classified to strong/moderate/weak + a tentative `conductor_like`. `reliability` is
    flagged 'low' -- feed it to diagnose_hypothesis as a PRIOR, not a determination.
    Thresholds are physical heuristics (not GT-tuned).
    """
    from gpr_data_processing.attributes.envelope import envelope

    frame = np.asarray(session.arrays[signed_handle], float)         # (n_samples, n_traces)
    ti, si = int(trace_idx), int(sample_idx)
    env = np.abs(envelope(frame, axis=0))[:, ti]
    n = env.size
    period = max(4, int(round(1.0 / (fc_hz * dt_ns * 1e-9))))        # ricker period in samples
    apex = float(env[si])
    noise = float(np.median(env)) + 1e-12
    snr_db = 20.0 * np.log10(apex / noise)
    dw = float(env[:max(period, si - 2 * period)].max()) if si > 2 * period else apex
    rel = apex / (dw + 1e-12)
    main = env[max(0, si - period // 2): si + period // 2 + 1]
    tail = env[min(n, si + period): min(n, si + 3 * period)]
    ringing = float(tail.mean() / (main.mean() + 1e-12)) if main.size and tail.size else 0.0

    if snr_db >= 20.0 and rel >= 0.30:
        strength = "strong"
    elif snr_db < 10.0 or rel < 0.10:
        strength = "weak"
    else:
        strength = "moderate"
    conductor_like = bool(strength == "strong" and ringing >= 0.40)
    return {"strength": strength, "conductor_like": conductor_like,
            "snr_db": round(snr_db, 1), "rel_to_direct": round(rel, 2), "ringing": round(ringing, 2),
            "reliability": "low -- relative amplitude only; gain/depth/attenuation/size/coupling "
                           "confound absolute amplitude in real data. Use as a weak prior.",
            "tool": "apex_amplitude"}

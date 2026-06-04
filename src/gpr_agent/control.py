"""A4 diagnostic CONTROL loop: diagnose an upstream problem -> act -> re-run.

The first (and most principled) action: a DEFOCUSED migrated image means the velocity
is wrong, so A4 triggers an AUTOFOCUS re-migration -- reusing GPR-Tools
`autofocus_velocity` + `focus_metric` (a focus metric, NOT ground truth -> compatible
with the no-calibration first principle). It also EARNS `velocity_calibrated=True`: the
velocity now comes from a physics focus sweep, not a nominal guess. Extensible to other
symptom->action pairs (clutter -> re-declutter, low SNR -> re-gain, tool mismatch ->
reroute). This is a SEPARATE orchestrator; the frozen deterministic oracle is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import tools
from .contracts import ImageRef, ImagingResult, InterpretationCase, Provenance, ToolCall
from .deterministic import LoopResult, fuse


@dataclass
class ControlAction:
    symptom: str
    diagnosis: str
    action: str
    metric_before: float
    metric_after: float


def _migrate_closure(pre: np.ndarray, dt_ns: float, dx_m: float, *, fc_hz, offset_m: float,
                     nz, zero_time: int, max_traces: int, mode: str):
    """Build migrate(v_m_per_s) -> migrated image (n_traces', nz), reusing the GPR-Tools
    Kirchhoff/LSM path (same setup as tools.image), so autofocus can sweep velocity."""
    from gpr_data_processing.migration.acquisition import AcquisitionGeometry, build_migrate_fn
    n_s, n_t = pre.shape
    dec = max(1, int(np.ceil(n_t / max_traces)))
    frame = np.ascontiguousarray(pre.T[::dec])
    dx_eff = dx_m * dec
    geo = AcquisitionGeometry.from_frame(frame=frame, dt_s=dt_ns * 1e-9, dx_m=dx_eff,
                                         fc_hz=fc_hz, offset_m=offset_m)
    out_nz = int(nz or n_s)
    mig = build_migrate_fn(geo, nz=out_nz, zero_time=int(zero_time), mode=mode)
    return (lambda v_ms: np.asarray(mig(frame, float(v_ms)))), frame.shape[0], out_nz, dx_eff


def autofocus_image(session, raw_handle: str, *, dt_ns: float, dx_m: float,
                    velocity_nominal: float = tools.NOMINAL_V, mode: str = "adjoint",
                    bgr: bool = True, fc_hz=None, offset_m: float = 0.1, nz=None,
                    zero_time: int = 0, max_traces: int = 220,
                    eps_bracket: tuple[float, float] = (4.0, 30.0), n_candidates: int = 11,
                    improve_frac: float = 0.05):
    """A1 imaging with a velocity-autofocus DIAGNOSIS. Returns (ImagingResult, ControlAction|None).

    Diagnose: is the nominal-velocity migration as focused as an autofocus sweep can get?
    Act (if a sweep beats nominal by `improve_frac`): re-migrate at the focusing velocity
    and mark `velocity_calibrated=True`. Otherwise keep nominal (no false calibration).
    """
    from gpr_data_processing.migration.autofocus import (
        autofocus_velocity, focus_metric, velocity_candidates_from_eps_r)

    raw = np.asarray(session.arrays[raw_handle], float)
    n_s, n_t = raw.shape
    pre = raw
    if bgr:
        from gpr_data_processing.preprocessing.bgr import background_removal
        pre = np.asarray(background_removal(raw, window=n_t), float)
    session.arrays["bscan"] = tools._envelope(pre, axis=0)
    bscan_ref = ImageRef(domain="radargram", handle="bscan", n_samples=n_s, n_traces=n_t, dt_ns=dt_ns, dx_m=dx_m)

    engine_mode = tools._MIGRATE_MODES.get(mode, mode)                       # 'adjoint' -> 'adjoint_only'
    migrate_at, n_tdec, out_nz, dx_eff = _migrate_closure(
        pre, dt_ns, dx_m, fc_hz=fc_hz, offset_m=offset_m, nz=nz, zero_time=zero_time,
        max_traces=max_traces, mode=engine_mode)
    focus_nom = focus_metric(migrate_at(velocity_nominal * 1e9), use_envelope=True)
    v_cands = velocity_candidates_from_eps_r(*eps_bracket, n_candidates)      # m/s
    res = autofocus_velocity(migrate_at, v_cands, metric="sharpness")
    focus_best = float(np.max(res.metric_curve))

    action = None
    if focus_best > focus_nom * (1.0 + improve_frac):                        # DIAGNOSE + ACT
        v_used = res.v_best / 1e9                                            # m/ns
        migrated = migrate_at(res.v_best)
        calibrated = True
        action = ControlAction(
            symptom="defocused migrated section",
            diagnosis=f"nominal v={velocity_nominal:g} m/ns under-focuses (sharpness {focus_nom:.3g})",
            action=f"autofocus re-migrate at v={v_used:.3f} m/ns",
            metric_before=round(focus_nom, 4), metric_after=round(focus_best, 4))
        note = (f"autofocus: re-migrated at v={v_used:.3f} m/ns "
                f"(sharpness {focus_nom:.3g} -> {focus_best:.3g}); velocity CALIBRATED by focus")
    else:                                                                    # keep nominal
        v_used = velocity_nominal
        migrated = migrate_at(velocity_nominal * 1e9)
        calibrated = False
        note = f"autofocus: nominal v={velocity_nominal:g} m/ns retained (no focus gain); velocity nominal"

    session.arrays["subsurface"] = tools._envelope(np.ascontiguousarray(migrated.T), axis=0)
    sub_ref = ImageRef(domain="subsurface", handle="subsurface", n_samples=out_nz,
                       n_traces=n_tdec, dt_ns=dt_ns, dx_m=dx_eff)
    if hasattr(session, "images"):
        session.images["bscan"] = bscan_ref
        session.images["subsurface"] = sub_ref
    imaging = ImagingResult(bscan=bscan_ref, subsurface=sub_ref, velocity_m_per_ns=v_used,
                            velocity_calibrated=calibrated,
                            provenance=Provenance(agent="A1",
                                                  tools=[ToolCall(tool="migrate_pylops", params={"mode": mode}),
                                                         ToolCall(tool="autofocus_velocity")], notes=note))
    return imaging, action


def orchestrate_with_control(session, raw_handle: str, *, dt_ns: float, dx_m: float,
                             velocity: float = tools.NOMINAL_V, max_rounds: int = 2):
    """A1(autofocus-controlled) -> A2 || A3 -> A4. Returns (LoopResult, [ControlAction])."""
    imaging, action = autofocus_image(session, raw_handle, dt_ns=dt_ns, dx_m=dx_m, velocity_nominal=velocity)
    v = imaging.velocity_m_per_ns
    ev_r = tools.analyze_radargram(session, imaging.bscan, velocity=v)
    ev_s = tools.analyze_subsurface(session, imaging.subsurface, velocity=v)
    hyps, queries = fuse(ev_r, ev_s, imaging)
    actions = [a for a in (action,) if a is not None]
    n_conf = sum(h.cross_domain_confirmed for h in hyps)
    case = InterpretationCase(
        hypotheses=hyps, follow_up_queries=queries, is_tentative=True,
        overall_note=(f"{n_conf} cross-domain-confirmed; "
                      + ("velocity autofocus-CALIBRATED -> depths now physical; "
                         if imaging.velocity_calibrated else "velocity nominal (uncalibrated); ")
                      + f"control actions: {[a.action for a in actions] or 'none'}"),
        provenance=Provenance(agent="A4", tools=[ToolCall(tool="fuse")],
                              notes=f"diagnostic control loop; {len(actions)} upstream action(s)"))
    return LoopResult(imaging=imaging, evidence=[ev_r, ev_s], case=case, rounds=len(actions)), actions

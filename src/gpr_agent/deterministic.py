"""Keyless deterministic engine: the full A1 -> (A2 || A3) -> A4 loop with NO LLM.

Runs the agent topology against real data using the shared `tools.py` detectors and
a deterministic A4 (`fuse`) that does the geometric work the prompts describe:
cross-domain association -> de-duplication -> falsification -> bounded re-plan. It
emits the same `contracts.InterpretationCase` the VLM A4 will, so the loop is
runnable + scoreable + reviewable today (no API key, no pydantic-ai), and serves as
the deterministic oracle/baseline the VLM path is compared against.

A4 here is geometry, not reasoning -- it associates detections, never invents them;
that is exactly the discipline the VLM A4 must keep.

=========================== FROZEN ORACLE -- DO NOT TUNE ===========================
This loop is a FIXED reference. Its parameters (below) are physically-motivated
defaults, NOT fit to any data. Do NOT change them to improve a benchmark score:
selecting a value by its result on a single line (n=1) is leakage/selection bias
(see AGENTS_DESIGN.md first principle, and no-calibration). Changing a value must be
a deliberate, documented decision evaluated on a held-out / multi-dataset split --
never the TU1208 number. `test_oracle_frozen.py` pins the behaviour so a casual edit
fails loudly. Tuning is a non-goal for this module.
====================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass

from . import tools
from .contracts import (
    Detection, DomainEvidence, Hypothesis, HypothesisStatus, ImagingResult,
    InterpretationCase, Provenance, Query, ToolCall,
)

# Canonical FROZEN parameters (single source of truth for documentation/audit).
# These are physical priors, not tuned values -- see the banner above.
FROZEN_PARAMS = {
    "velocity_m_per_ns": tools.NOMINAL_V,   # nominal; uncalibrated (velocity_calibrated=False)
    "A2_max_dets": 20, "A2_min_depth_m": 0.2,
    "A3_max_dets": 20, "A3_min_depth_m": 0.2, "A3_nbhd": 9, "A3_thresh_pct": 92.0,
    "fuse_dx_tol_m": 0.4, "fuse_dz_tol_m": 0.15,
    "max_rounds": 2, "reprobe_thresh_pct": 85.0,
    "migration": "adjoint", "declutter": "bgr",
}


@dataclass
class LoopResult:
    imaging: ImagingResult
    evidence: list[DomainEvidence]
    case: InterpretationCase
    rounds: int
    usage: object = None        # token usage for the VLM path (None for the keyless oracle)


def _is_boundary(depth_m: float, max_depth_m: float) -> bool:
    return depth_m >= 0.98 * max_depth_m


def _placeable(dets: list[Detection], v: float) -> list[Detection]:
    """Keep detections that can be placed in (x, depth); backfill depth from twtt.

    The deterministic tools always set x_m + depth_m, so this is a no-op for the
    oracle. VLM-produced detections may omit fields -> drop the unplaceable ones and
    derive depth = 0.5*v*twtt when only twtt is given.
    """
    out: list[Detection] = []
    for d in dets:
        depth = d.depth_m if d.depth_m is not None else (
            0.5 * v * d.twtt_ns if d.twtt_ns is not None else None)
        if depth is None or d.x_m is None:
            continue
        out.append(d if d.depth_m is not None else d.model_copy(update={"depth_m": round(depth, 3)}))
    return out


def _match(r: Detection, s: Detection, dx_tol: float, dz_tol: float) -> float:
    """Normalised (x, depth) distance if r and s could be the same target, else inf."""
    if r.x_m is None or s.x_m is None or r.depth_m is None or s.depth_m is None:
        return float("inf")
    dx, dz = abs(r.x_m - s.x_m) / dx_tol, abs(r.depth_m - s.depth_m) / dz_tol
    return (dx * dx + dz * dz) ** 0.5 if (dx <= 1.0 and dz <= 1.0) else float("inf")


def fuse(ev_r: DomainEvidence, ev_s: DomainEvidence, imaging: ImagingResult, *,
         dx_tol: float = 0.4, dz_tol: float = 0.15) -> tuple[list[Hypothesis], list[Query]]:
    """A4 geometry: associate across domains, de-dup, grade, and pose re-plan queries.

    One physical target = one Hypothesis even when it shows in both domains
    (cross-domain match -> a single `cross_domain_confirmed` hypothesis -- NO double
    count). Single-domain detections become `ambiguous` hypotheses + a falsifiable
    Query to the *other* domain.
    """
    v = imaging.velocity_m_per_ns or tools.NOMINAL_V
    max_depth = tools._depth_m(imaging.bscan.n_samples - 1, imaging.bscan.dt_ns or 1.0, v)
    # Robust to VLM-produced evidence (optional fields): keep only geometrically
    # placeable detections, backfilling depth from twtt when missing. No-op for the
    # deterministic tools (which always set x_m + depth_m) -> oracle behaviour unchanged.
    r_dets, s_dets = _placeable(ev_r.detections, v), _placeable(ev_s.detections, v)
    used_s: set[int] = set()
    hyps: list[Hypothesis] = []
    queries: list[Query] = []
    hid = 0

    # 1) greedy cross-domain association (strongest radargram apex first)
    for r in sorted(r_dets, key=lambda d: d.confidence, reverse=True):
        best_j, best_d = None, float("inf")
        for j, s in enumerate(s_dets):
            if j in used_s:
                continue
            d = _match(r, s, dx_tol, dz_tol)
            if d < best_d:
                best_j, best_d = j, d
        hid += 1
        if best_j is not None:                                  # CONFIRMED in both domains
            s = s_dets[best_j]; used_s.add(best_j)
            x = round((r.x_m + s.x_m) / 2, 3)
            z = round((r.depth_m + s.depth_m) / 2, 3)
            boundary = _is_boundary(z, max_depth)
            conf = round(min(0.95, (r.confidence + s.confidence) / 2 + 0.10), 3)
            hyps.append(Hypothesis(
                id=f"H{hid:03d}",
                statement=f"point reflector at x~{x:.2f} m, depth~{z:.2f} m "
                          f"(uncalibrated v={v:g} m/ns; corroborated in both domains)",
                status=HypothesisStatus.refuted if boundary else HypothesisStatus.supported,
                confidence=0.1 if boundary else conf, x_m=x, depth_m=z,
                predicted_in=["radargram", "subsurface"],
                supporting_evidence=[] if boundary else [r.id, s.id],
                refuting_evidence=[r.id, s.id] if boundary else [],
                cross_domain_confirmed=not boundary,
            ))
        else:                                                   # radargram-only -> ambiguous
            boundary = _is_boundary(r.depth_m, max_depth)
            hyps.append(Hypothesis(
                id=f"H{hid:03d}",
                statement=f"possible point reflector at x~{r.x_m:.2f} m, depth~{r.depth_m:.2f} m "
                          f"(radargram only; not yet corroborated in the subsurface)",
                status=HypothesisStatus.refuted if boundary else HypothesisStatus.ambiguous,
                confidence=0.1 if boundary else round(r.confidence * 0.8, 3),
                x_m=r.x_m, depth_m=r.depth_m, predicted_in=["radargram", "subsurface"],
                supporting_evidence=[] if boundary else [r.id],
                refuting_evidence=[r.id] if boundary else [],
            ))
            if not boundary:
                queries.append(Query(
                    domain="subsurface",
                    region=(r.x_m - dx_tol, r.depth_m - dz_tol, r.x_m + dx_tol, r.depth_m + dz_tol),
                    question=f"Is there a blob consistent with a point reflector near "
                             f"x={r.x_m:.2f} m, depth={r.depth_m:.2f} m?"))

    # 2) subsurface-only detections -> ambiguous hypotheses + a query to the radargram
    for j, s in enumerate(s_dets):
        if j in used_s:
            continue
        hid += 1
        boundary = _is_boundary(s.depth_m, max_depth)
        hyps.append(Hypothesis(
            id=f"H{hid:03d}",
            statement=f"possible reflector at x~{s.x_m:.2f} m, depth~{s.depth_m:.2f} m "
                      f"(subsurface only; no corroborating radargram apex)",
            status=HypothesisStatus.refuted if boundary else HypothesisStatus.ambiguous,
            confidence=0.1 if boundary else round(s.confidence * 0.8, 3),
            x_m=s.x_m, depth_m=s.depth_m, predicted_in=["radargram", "subsurface"],
            supporting_evidence=[] if boundary else [s.id],
            refuting_evidence=[s.id] if boundary else [],
        ))
        if not boundary:
            queries.append(Query(
                domain="radargram",
                region=(s.x_m - dx_tol, s.depth_m - dz_tol, s.x_m + dx_tol, s.depth_m + dz_tol),
                question=f"Is there a hyperbola apex consistent with a point reflector near "
                         f"x={s.x_m:.2f} m, depth={s.depth_m:.2f} m?"))

    hyps.sort(key=lambda h: h.confidence, reverse=True)
    return hyps, queries


def _reindex_subsurface(dets: list[Detection]) -> DomainEvidence:
    """Dedup (rounded x,depth) + fresh ids, so re-probed detections merge cleanly."""
    seen, keep = set(), []
    for d in sorted(dets, key=lambda d: d.confidence, reverse=True):
        key = (round(d.x_m or 0, 2), round(d.depth_m or 0, 2))
        if key in seen:
            continue
        seen.add(key); keep.append(d)
    out = [d.model_copy(update={"id": f"s{i:03d}"}) for i, d in enumerate(keep)]
    return DomainEvidence(domain="subsurface", detections=out, quality=f"{len(out)} blobs (merged after re-probe)",
                          provenance=Provenance(agent="A3", tools=[ToolCall(tool="maximum_filter")]))


def orchestrate_deterministic(session, raw_handle: str, *, dt_ns: float, dx_m: float,
                              velocity: float = tools.NOMINAL_V, max_rounds: int = 2,
                              reprobe_thresh_pct: float = 85.0,
                              migrate: str | bool = "adjoint", bgr: bool = True) -> LoopResult:
    """A1 -> (A2 || A3) -> A4 with a bounded re-plan. Returns imaging + evidence + case.

    `migrate`/`bgr` are A1's imaging choices (A3 just detects on the result): 'adjoint'
    runs real Kirchhoff migration so A3 sees a focused section; False keeps the cheap
    depth-relabelled envelope.
    """
    imaging = tools.image(session, raw_handle, dt_ns=dt_ns, dx_m=dx_m, velocity=velocity,
                          migrate=migrate, bgr=bgr)
    ev_r = tools.analyze_radargram(session, imaging.bscan, velocity=velocity)
    ev_s = tools.analyze_subsurface(session, imaging.subsurface, velocity=velocity)
    hyps, queries = fuse(ev_r, ev_s, imaging)

    rounds = 0
    while queries and rounds < max_rounds:
        sub_qs = [q for q in queries if q.domain == "subsurface"]   # re-probe subsurface only
        if not sub_qs:
            break
        rounds += 1
        extra: list[Detection] = list(ev_s.detections)
        for q in sub_qs:                                            # re-examine each region, more sensitively
            extra.extend(tools.analyze_subsurface(session, imaging.subsurface, velocity=velocity,
                                                  region=q.region, thresh_pct=reprobe_thresh_pct).detections)
        merged = _reindex_subsurface(extra)
        if len(merged.detections) == len(ev_s.detections):         # nothing new found -> stop
            break
        ev_s = merged
        hyps, queries = fuse(ev_r, ev_s, imaging)

    n_conf = sum(h.cross_domain_confirmed for h in hyps)
    n_amb = sum(h.status == HypothesisStatus.ambiguous for h in hyps)
    n_ref = sum(h.status == HypothesisStatus.refuted for h in hyps)
    case = InterpretationCase(
        hypotheses=hyps,
        follow_up_queries=queries,                                 # unresolved questions (never conclusions)
        overall_note=(f"{n_conf} cross-domain-confirmed, {n_amb} single-domain (ambiguous), "
                      f"{n_ref} refuted (boundary); {rounds} re-plan round(s). Velocity {velocity:g} m/ns "
                      f"is UNCALIBRATED, so depths and confidences are RELATIVE, not calibrated."),
        is_tentative=True,
        provenance=Provenance(agent="A4", tools=[ToolCall(tool="fuse", params={"max_rounds": max_rounds})],
                              notes="deterministic geometric fusion (no LLM)"),
    )
    return LoopResult(imaging=imaging, evidence=[ev_r, ev_s], case=case, rounds=rounds)

"""Polarity -> candidate material: the material-determination step (reads the KB).

Wraps `gpr_kb.reference.candidates_by_polarity` -- the single canonical source --
validated end-to-end on FDTD truth in `../run_polarity_material.py` (metal/void read
NEGATIVE, water reads POSITIVE; the true material lands in the candidate set).

Discipline (carried from the KB + no-calibration principle): polarity NARROWS the
candidate set, it does NOT uniquely identify the material (metal and air-void both
read NEGATIVE). Amplitude + depth disambiguate further. So this returns a *candidate
set* and a *tentative* best, never a determination claim.
"""
from __future__ import annotations

from typing import Optional

# Polarity (estimate_apex_polarity enum value / words / sign) -> reflection sign.
_SIGN = {
    "POSITIVE": 1, "positive": 1, "normal": 1, "+1": 1, 1: 1,
    "NEGATIVE": -1, "negative": -1, "reversed": -1, "-1": -1, -1: -1,
    "AMBIGUOUS": 0, "ambiguous": 0, "mixed": 0, "unknown": 0, "0": 0, 0: 0, None: 0,
}


def polarity_sign(polarity) -> int:
    """Map any polarity representation to +1 / -1 / 0."""
    return _SIGN.get(polarity, _SIGN.get(str(polarity), 0))


def candidates(polarity, eps_r_host: float, *, top: int = 5) -> list[tuple[str, dict]]:
    """Candidate buried-object types for an observed apex polarity in a host soil."""
    from gpr_kb import reference as ref
    return ref.candidates_by_polarity(polarity_sign(polarity), eps_r_host)[:top]


def best_material(polarity, eps_r_host: float) -> Optional[str]:
    """The single most-likely candidate id (conductors first for NEGATIVE), or None."""
    c = candidates(polarity, eps_r_host, top=1)
    return c[0][0] if c else None


def annotate_hypothesis(hypothesis, polarity, eps_r_host: float):
    """Set `hypothesis.material` to the best candidate + record the candidate set tentatively."""
    cands = candidates(polarity, eps_r_host, top=3)
    if not cands or polarity_sign(polarity) == 0:
        return hypothesis
    hypothesis.material = cands[0][0]
    ids = ", ".join(c[0] for c in cands)
    hypothesis.statement += f" | polarity {polarity}: candidate material(s) {ids} (not unique)"
    return hypothesis


def _kind_of(object_id: str) -> str:
    """Canonical object kind from the KB (was a keyword heuristic; now the one source)."""
    from gpr_kb import reference as ref
    return ref.kind_of(object_id)


def diagnose_hypothesis(hypothesis, polarity, eps_r_host: float, *, conductor: bool | None = None,
                        amplitude: str | None = None):
    """DIAGNOSE the object: set `hypothesis.kind` + `.material` from polarity (-> KB
    candidate object types) with depth as an honest note. An OPTIONAL amplitude cue
    (`conductor`, `amplitude`='strong'/'moderate'/'weak') disambiguates the
    polarity-degenerate set -- metal pipe vs air void vs plastic all read NEGATIVE.

    Graded + tentative (no-overclaim, no-calibration): polarity narrows to a candidate
    KIND set; the amplitude cue narrows further; depth outside the KB typical range is
    flagged, not used to exclude (the FDTD/dev pipes are deliberately shallow).
    """
    from gpr_kb import reference as ref
    if polarity_sign(polarity) == 0:
        return hypothesis
    cands = ref.candidates_by_polarity(polarity_sign(polarity), eps_r_host)
    if not cands:
        return hypothesis
    # AMPLITUDE IS UNRELIABLE (gain/depth/attenuation/size confounds) -> "do no harm":
    # only POSITIVE, high-confidence evidence narrows. A conductor *ringing* (conductor=True)
    # narrows to metal; absence of ringing (False) must NOT exclude metal. Only a confident
    # 'strong' narrows; 'weak' could be a deep/attenuated strong reflector, so it excludes nothing.
    narrow_conductor = True if conductor is True else None
    narrow_ampl = amplitude if amplitude == "strong" else None
    if narrow_conductor is not None or narrow_ampl is not None:
        cands = ref.disambiguate(cands, conductor=narrow_conductor, reflectivity=narrow_ampl)
    best_id, best = cands[0]
    kinds: list[str] = []
    for sid, _ in cands:                                               # distinct candidate KINDS (canonical)
        k = ref.kind_of(sid)
        if k != "unknown" and k not in kinds:
            kinds.append(k)
    hypothesis.kind = kinds[0] if kinds else ref.kind_of(best_id)
    hypothesis.material = best_id
    note = ""
    z = hypothesis.depth_m
    if z is not None and not (best["typical_depth_min_m"] <= z <= best["typical_depth_max_m"]):
        note = f", depth {z:.2f} m outside typical {best['typical_depth_min_m']}-{best['typical_depth_max_m']} m"
    cue = (f", amplitude cue(conductor={conductor}, refl={amplitude})"
           if (conductor is not None or amplitude is not None) else "")
    ambiguity = ("polarity alone cannot uniquely ID -- amplitude (conductor vs dielectric) + "
                 "shape/radius needed" if len(kinds) > 1 else "")
    hypothesis.statement += (f" | DIAGNOSIS: polarity {polarity}{cue} -> candidate kinds "
                             f"{{{', '.join(kinds) or '?'}}}, best {hypothesis.kind} [{best_id}]{note}"
                             + (f"; {ambiguity}" if ambiguity else ""))
    return hypothesis


def diagnose_case(case, evidence, eps_r_host: float):
    """Diagnose kind+material for every hypothesis, using its cited radargram detection's
    polarity (Detection.polarity, set by A2's check_polarity / tools.apex_polarity)."""
    pol = {d.id: d.polarity for e in evidence for d in e.detections
           if e.domain == "radargram" and d.polarity}
    for h in case.hypotheses:
        p = next((pol[e] for e in h.supporting_evidence if e in pol), None)
        if p:
            diagnose_hypothesis(h, p, eps_r_host)
    return case

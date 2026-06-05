"""Bridge the corpus generator's flat builder output to a domain-native SimulationCard.

The 7 builders emit flat ``_obj(...)`` dicts (kind + geometry params) + a single soil host. This
module maps those into the GPR-Sim domain model (SyntheticFeature -> ScenarioPrimitive, per-scene
spatial/construction relations, a HostProfile) and assembles a real SimulationCard via
subsurface_platform.synthetic.build_simulation_card. run_one writes the result as card.json next to
the (kept) flat labels.json -- so every synthetic scene is first-class in the authoritative model.

See docs/civil-host-and-standards-data-model.md (Phase 4).
"""
from __future__ import annotations

import sys

sys.path[:0] = [r"e:/github/GPR-Sim/src", r"e:/github/GPR-KnowledgeBase", r"e:/github/GPR-Tools/src"]

from subsurface_platform.domain import (                       # noqa: E402
    Horizon,
    HorizonRole,
    HostContext,
    HostProfile,
    KnownFaciesType,
)
from subsurface_platform.domain.primitive_geometry import (    # noqa: E402
    BoxGeometry,
    CylinderGeometry,
    PrismGeometry,
    SphereGeometry,
)
from subsurface_platform.domain.acquisition_priors import AcquisitionPriors  # noqa: E402
from subsurface_platform.domain.scenario_relation import ScenarioRelationType  # noqa: E402
from subsurface_platform.synthetic import SyntheticFeature, build_simulation_card  # noqa: E402

_Y = 0.20  # nominal cross-survey extent (m) for geometry metadata (the corpus is a 2-D x-z profile)
_K = KnownFaciesType

# kind -> (feature_type, klass, is_target, facies). Geology (boulder/root) is non-target clutter;
# the topsoil cap is context (not a target). Pipes/conduits/rebar/slab/trench are inspection features.
KIND_MAP: dict[str, tuple[str, str, bool, KnownFaciesType | None]] = {
    "pipe": ("pipe", "utility", True, _K.DIFFRACTION_HYPERBOLA),
    "conduit": ("conduit", "utility", True, _K.DIFFRACTION_HYPERBOLA),
    "pipe_void": ("pipe_void", "utility", True, _K.DIFFRACTION_HYPERBOLA),
    "duct_bank_envelope": ("duct_envelope", "structural", True, _K.PLANAR_REFLECTION),
    "protective_slab": ("slab", "structural", True, _K.PLANAR_REFLECTION),
    "rebar": ("rebar", "structural", True, _K.DIFFRACTION_HYPERBOLA),
    "trench_backfill": ("trench", "structural", True, _K.PLANAR_REFLECTION),
    "topsoil_cap": ("topsoil_cap", "structural", False, None),
    "tree_root": ("tree_root", "geological", False, _K.VOLUMETRIC_SCATTER),
    "boulder": ("boulder", "geological", False, _K.DIFFRACTION_HYPERBOLA),
}


def _geometry_from_obj(obj: dict):
    """Best-effort flat-obj -> PrimitiveGeometry (None on any mismatch). x=survey, z=depth, y=cross."""
    try:
        kind = obj.get("kind", "")
        if "polygon_m" in obj:
            pts = [(float(px), float(pz)) for px, pz in obj["polygon_m"]]
            return PrismGeometry(polygon_xz=pts, y_lo=0.0, y_hi=_Y)
        if "box_m" in obj:
            b = obj["box_m"]
            return BoxGeometry(x_lo=float(b["x_min"]), x_hi=float(b["x_max"]),
                               z_lo=float(b["depth_top"]), z_hi=float(b["depth_bottom"]),
                               y_lo=0.0, y_hi=_Y)
        if "center_x_m" in obj:
            cx, dz, r = float(obj["center_x_m"]), float(obj["depth_m"]), float(obj["radius_m"])
            if kind == "boulder":
                return SphereGeometry(center_xyz=(cx, _Y / 2, dz), radius_m=r)
            return CylinderGeometry(axis_start_xyz=(cx, 0.0, dz), axis_end_xyz=(cx, _Y, dz), radius_m=r)
    except Exception:
        return None
    return None


def features_from_objs(objs: list[dict]) -> list[SyntheticFeature]:
    """Map flat builder objects to typed SyntheticFeatures (stable ids feat_<kind>_<n>)."""
    feats: list[SyntheticFeature] = []
    counts: dict[str, int] = {}
    for obj in objs:
        kind = obj.get("kind", "object")
        ft, klass, target, facies = KIND_MAP.get(kind, (kind, "geological", False, None))
        n = counts.get(kind, 0); counts[kind] = n + 1
        feats.append(SyntheticFeature(
            feature_id=f"feat_{kind}_{n}", feature_type=ft, klass=klass, target=target,
            material=obj.get("material", "air"), facies=facies, geometry=_geometry_from_obj(obj),
            name=f"{ft}_{n}"))
    return feats


def _ids_by_type(feats: list[SyntheticFeature], ft: str) -> list[str]:
    return [f.feature_id for f in feats if f.feature_type == ft]


def relations_for_scene(scene_type: str, feats: list[SyntheticFeature]) -> list[tuple]:
    """Per-archetype structural relations (the spatial/construction binding -> composite)."""
    rels: list[tuple] = []
    trench = _ids_by_type(feats, "trench")
    caps = _ids_by_type(feats, "topsoil_cap")
    pipes = _ids_by_type(feats, "pipe")
    slabs = _ids_by_type(feats, "slab")
    envelopes = _ids_by_type(feats, "duct_envelope")
    conduits = _ids_by_type(feats, "conduit")
    if scene_type == "utility_trench" and trench:
        for p in pipes:                                          # bedded pipe inside the trench
            rels.append((p, ScenarioRelationType.SPATIAL_CONTAINED_IN, trench[0]))
        for c in caps:                                           # resurfaced trench cap
            rels.append((c, ScenarioRelationType.CONSTRUCTION_RESURFACED_AFTER, trench[0]))
    elif scene_type == "protective_concrete":
        for s in slabs:
            for p in pipes:                                      # slab protecting/over the pipe
                rels.append((s, ScenarioRelationType.SPATIAL_ABOVE, p))
    elif scene_type == "duct_bank" and envelopes:
        for c in conduits:                                       # conduits within the envelope
            rels.append((c, ScenarioRelationType.SPATIAL_CONTAINED_IN, envelopes[0]))
    # single_pipe / tree_roots / boulder_field / rebar_mesh -> no structural relation (simple)
    return rels


def host_profile_from_soil(soil: str) -> HostProfile:
    """Wrap the corpus single-soil host as a bare_soil HostProfile (additive; civil contexts come
    from NL resolution in Phase 5)."""
    return HostProfile(context=HostContext.BARE_SOIL, default_coupling="ground_coupled",
                       layers=[Horizon(role=HorizonRole.NATURAL_SOIL, material=soil)],
                       plausible_targets=["utility_pipe", "trench", "boulder", "void"],
                       standard_refs=["USCS ASTM D2487"])


def build_card_for_scene(*, card_id: str, scene_type: str, soil: str, objs: list[dict],
                         coupling: str | None, note: str):
    """Assemble the domain-native SimulationCard (+ primitives) for one generated scene."""
    feats = features_from_objs(objs)
    rels = relations_for_scene(scene_type, feats)
    host = host_profile_from_soil(soil)
    acq = AcquisitionPriors(coupling=coupling) if coupling else AcquisitionPriors(coupling=host.default_coupling)
    summary = f"{scene_type} in {soil}: " + ", ".join(f.feature_type for f in feats)
    card, prims = build_simulation_card(
        card_id=card_id, host_profile=host, features=feats, relations=rels,
        acquisition_priors=acq, summary=summary[:300],
        tested_claim=f"A {scene_type} scene produces its characteristic GPR signature.",
        narrative=note or summary[:300])
    return card, prims

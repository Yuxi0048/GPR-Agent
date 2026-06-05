"""Overnight subsurface-scene corpus generator: parametric scene -> gprMax B-scan + GT.

Drives ``gpr_agent.sim_scenes`` (build123d CAD -> OCP voxelize -> gprMax HDF5 -> B-scan
on GPU via GPR-Sim's runtime). Each sample saves the gprMax INPUT (geometry.h5 +
materials.txt + per-trace t*.in) and OUTPUT (per-trace t*.out + bscan.npy + preview),
plus a GT ``labels.json`` (every object: kind, material, position, size, eps_r) and a
``provenance.json`` (params, seed, timestamp, engine). One ``manifest.jsonl`` line per
sample. Robust by design: each scene is isolated in try/except, failures are logged and
the loop continues, so it accumulates labeled data unattended.

Scene types (from the subsurface_model_corpus):
  utility   : single_pipe, duct_bank, utility_trench, protective_concrete
  ambiguity : tree_roots (root mimics small pipe/void), boulder_field (point diffractor),
              rebar_mesh (periodic hyperbolas)

Run:
    python scripts/generate_subsurface_corpus.py [--hours 9] [--max 400] [--seed 0]
needs PYTHONPATH = GPR-Agent/src;GPR-Sim/src;GPR-KnowledgeBase;GPR-Tools/src;GPR-Interpretation/src
  (GPR-Interpretation/src provides gpr_reasoning, pulled in by `import gpr_agent`)
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gpr_agent import sim_scenes as S
from subsurface_platform.domain.host_correlation import (
    scene_host_coupling, correlated_shape_for, material_for, burial_depth_for)
from gpr_data_processing.footprint import footprint_half_width_m   # footprint physics (single source)
from subsurface_platform.domain.acquisition_coverage import coverage as _coverage   # domain preflight (DI)

OUT_ROOT = Path("e:/github/GPR-Sim/data/generated_corpus")
SOILS = ["dry_sand", "dry_clay", "moist_limestone", "saturated_sand", "wet_clay", "silt", "loam"]
PIPE_MATS = ["steel", "cast_iron", "pvc", "hdpe", "concrete"]   # real names; em_spec maps metals->PEC
SCENE_WEIGHTS = {                       # how often each type is drawn
    "single_pipe": 3, "duct_bank": 2, "utility_trench": 2,
    "protective_concrete": 2, "tree_roots": 2, "boulder_field": 2, "rebar_mesh": 1,
}

# Host-correlation policy -----------------------------------------------------------------------
# The plausible host set per scene type is READ from the single source of truth,
# subsurface_platform.domain.host_correlation.CORRELATION_TABLE (Step 2 of
# GPR-Workbench/docs/coupling-revision-plan.md) -- no local copy. A type absent from the derived map
# is host-agnostic (falls back to SOILS in choose_host). Edit the SETS in host_correlation.py
# (domain knowledge); the correlation STRENGTH stays an operator input (--realistic-host-prob),
# never a platform constant.
HOST_COUPLING = scene_host_coupling()


def choose_host(scene_type: str, host_rng, realistic_prob: float) -> tuple[str, bool, list[str]]:
    """Pick the scene host under the decorrelation policy.

    With probability ``realistic_prob`` the host is drawn from the type's plausible set
    (``HOST_COUPLING``); otherwise from ALL hosts (``SOILS``) -- breaking the host->type shortcut
    so a model cannot infer the object from the background. ``realistic_prob`` is the operator's
    per-run control, not a platform constant. Returns (host, drawn_realistic, natural_set).
    """
    natural = HOST_COUPLING.get(scene_type) or SOILS
    drawn_realistic = bool(host_rng.random() < realistic_prob)
    host = str(host_rng.choice(natural if drawn_realistic else SOILS))
    return host, drawn_realistic, list(natural)


def _eps(mat: str) -> float:
    try:
        return float(S.em_spec(mat)["eps_r"])
    except Exception:
        return float("nan")


def _is_conductor(mat: str) -> bool:
    return mat.strip().lower() in ("pec", "metal", "steel", "cast_iron", "lead",
                                   "paper_insulated_lead_covered", "pilc")


def _obj(kind, material, *, x=None, depth=None, radius=None, box=None, polygon=None, ambiguity=False):
    d = {"kind": kind, "material": material, "eps_r": round(_eps(material), 3),
         "conductor": _is_conductor(material), "ambiguity": ambiguity}
    if x is not None:
        d.update(center_x_m=round(x, 4), depth_m=round(depth, 4), radius_m=round(radius, 4))
    if box is not None:
        d["box_m"] = {k: round(v, 4) for k, v in box.items()}
    if polygon is not None:
        d["polygon_m"] = [[round(px, 4), round(pz, 4)] for px, pz in polygon]
    return d


def _meta(stype, soil, soil_depth, *, fc=8e8, n_traces=40, ambiguity=False, note=""):
    v = 3e8 / math.sqrt(max(_eps(soil), 1.0))                 # host wave speed (m/s)
    tw = 2.2 * (soil_depth + 0.12) / v + 3e-9                 # two-way + margin
    return {"scene_type": stype, "host_material": soil, "host_eps_r": round(_eps(soil), 3),
            "fc_hz": fc, "n_traces": n_traces, "dx_m": 0.005,
            "time_window_s": float(min(max(tw, 1.0e-8), 2.2e-8)),
            "ambiguity": ambiguity, "note": note}


def _coverage_targets(objs):
    """(center_x_m, depth_m, half_extent_m) per target for the scan-coverage preflight.
    Skips the full-width topsoil cap (context, not a localized target)."""
    out = []
    for o in objs:
        if o.get("kind") == "topsoil_cap":
            continue
        if "center_x_m" in o:
            out.append((float(o["center_x_m"]), float(o["depth_m"]), float(o.get("radius_m", 0.0))))
        elif "box_m" in o:
            b = o["box_m"]
            out.append((0.5 * (b["x_min"] + b["x_max"]), 0.5 * (b["depth_top"] + b["depth_bottom"]),
                        0.5 * (b["x_max"] - b["x_min"])))
        elif "polygon_m" in o:
            xs = [p[0] for p in o["polygon_m"]]; zs = [p[1] for p in o["polygon_m"]]
            out.append((0.5 * (min(xs) + max(xs)), 0.5 * (min(zs) + max(zs)), 0.5 * (max(xs) - min(xs))))
    return out


# --------------------------------------------------------------------------- #
# Scene builders -> (Scene, list[label objects], meta)
# --------------------------------------------------------------------------- #
def build_single_pipe(rng, host_soil=None, setting="modern",
                       material_realistic=True, depth_realistic=True):
    # Correlation rows (domain host_correlation, strength-gated): pipe MATERIAL <-> installation era
    # (old_urban -> cast iron / AC; modern -> plastic; AWWA / ter Huurne 2024) and burial DEPTH <->
    # ground (frost-susceptible fine soils deeper, rock shallow; ASCE 32). host_soil=None / defaults
    # -> legacy behavior for promoted templates.
    soil = host_soil or rng.choice(SOILS); W = rng.uniform(1.0, 1.4); D = rng.uniform(0.65, 0.9)
    sc = S.Scene(width_m=W, soil_depth_m=D, dx_m=0.005, soil_material=soil)
    x = rng.uniform(0.35, W - 0.35); r = rng.uniform(0.03, 0.07)
    mat = material_for("single_pipe", setting, rng, realistic=material_realistic)
    z = burial_depth_for(soil, rng, 0.20, D - 0.12, realistic=depth_realistic)
    sc.add_pipe(center_x_m=x, depth_m=z, radius_m=r, material=mat)
    objs = [_obj("pipe", mat, x=x, depth=z, radius=r)]
    if mat in ("pvc", "hdpe", "poly_ethylene", "concrete") and rng.random() < 0.6:   # empty -> inner air
        sc.add_void(center_x_m=x, depth_m=z, radius_m=r * 0.7, material="air")
        objs.append(_obj("pipe_void", "air", x=x, depth=z, radius=r * 0.7))
    return sc, objs, _meta("single_pipe", soil, D, note=f"{mat} pipe ({setting}) @ {z:.2f}m")


def build_duct_bank(rng):
    soil = rng.choice(SOILS); W = rng.uniform(1.2, 1.6); D = rng.uniform(0.7, 0.95)
    sc = S.Scene(width_m=W, soil_depth_m=D, dx_m=0.005, soil_material=soil)
    cols = rng.integers(2, 4); rows = rng.integers(1, 3)
    cw, ch = 0.10, 0.10                                                  # conduit cell
    bw = cols * cw + 0.06; bh = rows * ch + 0.06
    cx = rng.uniform(bw / 2 + 0.1, W - bw / 2 - 0.1); top = rng.uniform(0.18, 0.35)
    sc.add_box(x_min_m=cx - bw / 2, x_max_m=cx + bw / 2, depth_top_m=top,
               depth_bottom_m=top + bh, material="concrete", name="duct_bank_envelope")
    objs = [_obj("duct_bank_envelope", "concrete",
                 box={"x_min": cx - bw / 2, "x_max": cx + bw / 2, "depth_top": top, "depth_bottom": top + bh})]
    cmat = rng.choice(["pvc", "air"])
    for i in range(int(cols)):
        for j in range(int(rows)):
            ox = cx - bw / 2 + 0.05 + i * cw + cw / 2
            oz = top + 0.05 + j * ch + ch / 2
            sc.add_void(center_x_m=ox, depth_m=oz, radius_m=0.03, material=cmat)
            objs.append(_obj("conduit", cmat, x=ox, depth=oz, radius=0.03))
    return sc, objs, _meta("duct_bank", soil, D, n_traces=44,
                           note=f"{cols}x{rows} conduits in concrete")


def build_utility_trench(rng, host_soil=None, shape_realistic=True):
    # Step 3/4: the trench is dug in `native` (the decorrelated scene host when supplied) and its wall
    # angle CORRELATES with that soil -- cohesive (clay) stands vertical (box); granular (sand) must
    # slope (trapezoid / v_shape). Rule + soil->texture live in the domain (OSHA 1926 Subpart P /
    # Eurocode 7). shape_realistic gates that correlation under the SAME --realistic-host-prob strength
    # (False -> an off-distribution shape regardless of soil). host_soil=None -> legacy random native.
    native = host_soil or rng.choice(["wet_clay", "dry_clay", "moist_limestone"])
    backfill = rng.choice(["dry_sand", "gravel", "silt"])
    W = rng.uniform(1.1, 1.5); D = rng.uniform(0.7, 0.95)
    sc = S.Scene(width_m=W, soil_depth_m=D, dx_m=0.005, soil_material=native)
    tw_ = rng.uniform(0.35, 0.6); cx = rng.uniform(tw_ / 2 + 0.15, W - tw_ / 2 - 0.15)
    bottom = rng.uniform(0.5, 0.75)
    htop = tw_ / 2
    shape = correlated_shape_for("utility_trench", native, rng, realistic=shape_realistic)  # OSHA/EC7, strength-gated
    if shape == "box":
        sc.add_box(x_min_m=cx - htop, x_max_m=cx + htop, depth_top_m=0.0,
                   depth_bottom_m=bottom, material=backfill, name="trench_backfill")
        objs = [_obj("trench_backfill", backfill,
                     box={"x_min": cx - htop, "x_max": cx + htop, "depth_top": 0.0, "depth_bottom": bottom})]
    elif shape == "trapezoid":
        hbot = htop * rng.uniform(0.45, 0.8)                         # narrower floor
        corners = [(cx - htop, 0.0), (cx + htop, 0.0), (cx + hbot, bottom), (cx - hbot, bottom)]
        sc.add_polygon(corners_xz=corners, material=backfill, name="trench_backfill")
        objs = [_obj("trench_backfill", backfill, polygon=corners,
                     box={"x_min": cx - htop, "x_max": cx + htop, "depth_top": 0.0, "depth_bottom": bottom})]
    else:                                                            # v_shape: triangle converging to an apex
        corners = [(cx - htop, 0.0), (cx + htop, 0.0), (cx, bottom)]
        sc.add_polygon(corners_xz=corners, material=backfill, name="trench_backfill")
        objs = [_obj("trench_backfill", backfill, polygon=corners,
                     box={"x_min": cx - htop, "x_max": cx + htop, "depth_top": 0.0, "depth_bottom": bottom})]
    if shape != "v_shape" and rng.random() < 0.7:                    # bedded pipe at the (flat) trench bottom
        mat = rng.choice(["pvc", "steel", "concrete"]); r = rng.uniform(0.03, 0.06)
        sc.add_pipe(center_x_m=cx, depth_m=bottom - 0.08, radius_m=r, material=mat)
        objs.append(_obj("pipe", mat, x=cx, depth=bottom - 0.08, radius=r))
    note = f"{shape} {backfill} trench in {native}"
    if rng.random() < 0.5:                                           # resurfaced trench: full-width topsoil cap on top
        ts = rng.uniform(0.08, 0.16); ts_mat = rng.choice(["topsoil_moist", "loam"])
        sc.add_layer(depth_top_m=0.0, thickness_m=ts, material=ts_mat)   # painted last -> caps the trench
        objs.append(_obj("topsoil_cap", ts_mat,
                         box={"x_min": 0.0, "x_max": round(W, 4), "depth_top": 0.0, "depth_bottom": round(ts, 4)}))
        note += f" + {ts_mat} topsoil cap"
    return sc, objs, _meta("utility_trench", native, D, note=note)


def build_protective_concrete(rng):
    soil = rng.choice(SOILS); W = rng.uniform(1.1, 1.5); D = rng.uniform(0.7, 0.95)
    sc = S.Scene(width_m=W, soil_depth_m=D, dx_m=0.005, soil_material=soil)
    slab_top = rng.uniform(0.12, 0.28); slab_h = rng.uniform(0.08, 0.15)
    sw = rng.uniform(0.6, min(1.0, W - 0.2)); cx = rng.uniform(sw / 2 + 0.1, W - sw / 2 - 0.1)
    sc.add_box(x_min_m=cx - sw / 2, x_max_m=cx + sw / 2, depth_top_m=slab_top,
               depth_bottom_m=slab_top + slab_h, material="concrete", name="protective_slab")
    objs = [_obj("protective_slab", "concrete",
                 box={"x_min": cx - sw / 2, "x_max": cx + sw / 2, "depth_top": slab_top, "depth_bottom": slab_top + slab_h})]
    mat = rng.choice(["steel", "pvc", "concrete"]); pz = slab_top + slab_h + rng.uniform(0.12, 0.3)
    r = rng.uniform(0.04, 0.07)
    sc.add_pipe(center_x_m=cx + rng.uniform(-0.1, 0.1), depth_m=pz, radius_m=r, material=mat)
    objs.append(_obj("pipe", mat, x=cx, depth=pz, radius=r))
    return sc, objs, _meta("protective_concrete", soil, D, note="slab over pipe")


def build_tree_roots(rng):
    soil = rng.choice(["dry_sand", "loam", "silt", "topsoil_moist", "dry_clay"])
    W = rng.uniform(1.1, 1.5); D = rng.uniform(0.6, 0.85)
    sc = S.Scene(width_m=W, soil_depth_m=D, dx_m=0.005, soil_material=soil)
    n = int(rng.integers(3, 8)); objs = []
    for _ in range(n):
        x = rng.uniform(0.25, W - 0.25); z = rng.uniform(0.12, 0.45); r = rng.uniform(0.012, 0.03)
        sc.add_pipe(center_x_m=x, depth_m=z, radius_m=r, material="root")
        objs.append(_obj("tree_root", "root", x=x, depth=z, radius=r, ambiguity=True))
    return sc, objs, _meta("tree_roots", soil, D, ambiguity=True,
                           note=f"{n} roots mimic small pipes/voids")


def build_boulder_field(rng):
    soil = rng.choice(SOILS); W = rng.uniform(1.1, 1.5); D = rng.uniform(0.65, 0.9)
    sc = S.Scene(width_m=W, soil_depth_m=D, dx_m=0.005, soil_material=soil)
    n = int(rng.integers(1, 4)); objs = []
    rock = rng.choice(["granite", "limestone", "moist_limestone"])
    for _ in range(n):
        x = rng.uniform(0.25, W - 0.25); z = rng.uniform(0.2, 0.55); r = rng.uniform(0.04, 0.11)
        sc.add_pipe(center_x_m=x, depth_m=z, radius_m=r, material=rock)
        objs.append(_obj("boulder", rock, x=x, depth=z, radius=r, ambiguity=True))
    return sc, objs, _meta("boulder_field", soil, D, ambiguity=True,
                           note=f"{n} {rock} cobbles -> point diffractors")


def build_rebar_mesh(rng):
    soil = rng.choice(["concrete", "dry_sand", "moist_limestone"])
    W = rng.uniform(1.0, 1.4); D = rng.uniform(0.5, 0.75)
    sc = S.Scene(width_m=W, soil_depth_m=D, dx_m=0.005, soil_material=soil)
    n = int(rng.integers(4, 8)); spacing = (W - 0.3) / (n - 1); z = rng.uniform(0.1, 0.25)
    objs = []
    for i in range(n):
        x = 0.15 + i * spacing
        sc.add_pipe(center_x_m=x, depth_m=z, radius_m=0.01, material="steel")
        objs.append(_obj("rebar", "steel", x=x, depth=z, radius=0.01, ambiguity=True))
    return sc, objs, _meta("rebar_mesh", soil, D, ambiguity=True,
                           note=f"{n} bars @ {spacing*100:.0f} cm -> periodic hyperbolas")


BUILDERS = {
    "single_pipe": build_single_pipe, "duct_bank": build_duct_bank,
    "utility_trench": build_utility_trench, "protective_concrete": build_protective_concrete,
    "tree_roots": build_tree_roots, "boulder_field": build_boulder_field,
    "rebar_mesh": build_rebar_mesh,
}


def load_promoted_builders() -> dict:
    """Tier-T promoted templates: discover build_*.py in promoted_templates/ (origin=
    promoted_from_freeform). These graduated from a human-reviewed freeform proposal via
    corpus_tiers.py. See docs/corpus-two-tier-design.md. Best-effort: a broken stub is skipped."""
    import importlib.util
    out: dict = {}
    pdir = Path(__file__).resolve().parent / "promoted_templates"
    if not pdir.exists():
        return out
    for f in sorted(pdir.glob("build_*.py")):
        try:
            spec = importlib.util.spec_from_file_location(f.stem, f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)              # type: ignore[union-attr]
            fn = getattr(mod, f.stem, None)           # build_<name>
            if callable(fn):
                out[f.stem[len("build_"):]] = fn
        except Exception:
            continue
    return out


def all_builders() -> dict:
    """The live Tier-T set: natives + finalized promoted templates."""
    return {**BUILDERS, **load_promoted_builders()}


def _save_preview(png: Path, bscan: np.ndarray):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        b = np.asarray(bscan, float)
        g = b / (np.percentile(np.abs(b), 99, axis=0, keepdims=True) + 1e-12)   # per-trace AGC
        plt.figure(figsize=(5, 4))
        plt.imshow(np.clip(g, -1, 1), aspect="auto", cmap="gray")
        plt.xlabel("trace"); plt.ylabel("sample"); plt.tight_layout()
        plt.savefig(png, dpi=90); plt.close()
    except Exception:
        pass


def _clean_caches(out_dir: Path):
    import shutil
    for c in ("_cuda_cache", "_pycuda_cache", "_runtime_tmp"):
        shutil.rmtree(out_dir / c, ignore_errors=True)


# Scan trajectory: the survey line runs ALONG the scene width. The antenna steps from
# `SCAN_MARGIN_M` to `width - SCAN_MARGIN_M` at a fixed trace spacing, so every object
# (placed within [0.25, width-0.25]) is imaged -- not the old fixed [0.10, 0.49] window.
SCAN_MARGIN_M = 0.10
SCAN_STEP_M = 0.02            # trace spacing (good lateral hyperbola sampling at 800 MHz)


def run_one(scene_type: str, idx: int, seed: int, log, realistic_host_prob: float,
            n_traces_override: int = 0):
    rng = np.random.default_rng(seed)
    # Host-correlation policy: decorrelate object<->host with the operator-set probability. The host
    # is decided FIRST (a SEPARATE RNG seeded from `seed`, so object/clutter draws stay byte-identical
    # regardless of the prob), so utility_trench can match its wall angle to the soil (Step 3).
    host_rng = np.random.default_rng([seed, 0x484F5354])           # "HOST"
    host, drawn_realistic, natural = choose_host(scene_type, host_rng, realistic_host_prob)
    _builders = all_builders()
    # Per-mechanism realistic flags, each an independent draw at the SAME strength (host_rng, a
    # separate stream so object/geometry draws stay byte-stable). None where not applicable.
    shape_realistic = material_realistic = depth_realistic = None
    setting = None
    if scene_type == "utility_trench":
        shape_realistic = bool(host_rng.random() < realistic_host_prob)   # shape <-> soil (OSHA/EC7)
        sc, objs, meta = _builders[scene_type](rng, host_soil=host, shape_realistic=shape_realistic)
    elif scene_type == "single_pipe":
        setting = str(host_rng.choice(["old_urban", "modern", "greenfield"]))
        material_realistic = bool(host_rng.random() < realistic_host_prob)  # material <-> era
        depth_realistic = bool(host_rng.random() < realistic_host_prob)     # depth <-> ground
        sc, objs, meta = _builders[scene_type](rng, host_soil=host, setting=setting,
                                               material_realistic=material_realistic,
                                               depth_realistic=depth_realistic)
    else:
        sc, objs, meta = _builders[scene_type](rng)
    sc.soil_material = host
    meta["host_material"] = host
    meta["host_eps_r"] = round(_eps(host), 3)
    _v = 3e8 / math.sqrt(max(_eps(host), 1.0))                     # recompute host-dependent window
    meta["time_window_s"] = float(min(max(2.2 * (sc.soil_depth_m + 0.12) / _v + 3e-9, 1.0e-8), 2.2e-8))
    host_coupling = {"realistic_prob": realistic_host_prob, "drawn_realistic": drawn_realistic,
                     "decorrelated": not drawn_realistic, "natural_hosts": natural,
                     "shape_realistic": shape_realistic,        # trench shape <-> soil (None=N/A)
                     "setting": setting,                        # single_pipe installation era
                     "material_realistic": material_realistic,  # pipe material <-> era
                     "depth_realistic": depth_realistic}        # pipe depth <-> ground
    # Heterogeneous background soil: a correlated random eps_r field (realistic clutter).
    het = {"eps_spread_frac": 0.12, "correlation_length_m": float(rng.uniform(0.06, 0.15)),
           "n_levels": 9, "seed": seed}
    sc.soil_heterogeneity = het
    span = max(sc.width_m - 2 * SCAN_MARGIN_M, SCAN_STEP_M)
    n_traces = n_traces_override or max(24, int(round(span / SCAN_STEP_M)) + 1)
    scan_start, scan_step, tw_s = SCAN_MARGIN_M, SCAN_STEP_M, meta["time_window_s"]
    # Footprint-based scan-coverage PRE-FLIGHT (GPR-Tools): ensure every target's COMPLETE signature
    # is captured -- lateral (scan reaches the footprint edges), temporal (trace long enough for the
    # hyperbola wings), spatial (trace spacing below the anti-alias limit). Extend within the domain.
    cov_targets = _coverage_targets(objs)
    if cov_targets and not n_traces_override:
        pre = _coverage(cov_targets, scan_x_min_m=scan_start,
                        scan_x_max_m=scan_start + (n_traces - 1) * scan_step,
                        host_eps_r=meta["host_eps_r"], fc_hz=meta["fc_hz"],
                        footprint_fn=footprint_half_width_m, n_traces=n_traces, time_window_ns=tw_s * 1e9)
        if not pre["all_covered"]:
            scan_start = round(max(0.03, pre["recommended_scan_x_min_m"]), 4)
            scan_end = round(min(sc.width_m - 0.03, pre["recommended_scan_x_max_m"]), 4)
            n_traces = max(n_traces, int(round((scan_end - scan_start) / SCAN_STEP_M)) + 1,
                           int(pre.get("recommended_n_traces", n_traces)))
            scan_step = (scan_end - scan_start) / max(n_traces - 1, 1)
            tw_s = float(min(3.0e-8, max(tw_s, pre["recommended_time_window_ns"] * 1e-9 * 1.1)))
            meta["time_window_s"] = tw_s
    meta["n_traces"] = n_traces
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = OUT_ROOT / scene_type / f"{scene_type}_{ts}_{idx:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = S.run_bscan(sc, out_dir=out_dir, fc_hz=meta["fc_hz"], n_traces=n_traces,
                    scan_start_m=scan_start, scan_step_m=scan_step,
                    time_window_s=tw_s, gpu=True)
    dt_ns, dx_m, b = r["dt_ns"], r["dx_m"], r["bscan"]
    # Post-run coverage with the ACTUAL dt / n_samples / traces (records what the output really covers).
    scan_cov = None
    if cov_targets:
        scan_cov = _coverage(cov_targets, scan_x_min_m=scan_start,
                             scan_x_max_m=scan_start + (b.shape[1] - 1) * dx_m,
                             host_eps_r=meta["host_eps_r"], fc_hz=meta["fc_hz"],
                             footprint_fn=footprint_half_width_m, n_traces=b.shape[1],
                             time_window_ns=dt_ns * b.shape[0], dt_ns=dt_ns)
        if not scan_cov["all_covered"]:
            log(f"  WARN coverage {out_dir.name}: lat={scan_cov['all_lateral_covered']} "
                f"temp={scan_cov.get('all_temporal_covered')} aa={scan_cov.get('antialias_ok')} "
                f"(domain may be too narrow / cap hit)")
    _save_preview(out_dir / "bscan.png", b)
    # Domain-native artifact (Phase 4): lift the scene into a SimulationCard with typed features,
    # spatial/construction relations, a HostProfile and a derived composition -- written as card.json
    # next to the flat labels.json. Best-effort so a domain-import hiccup never fails generation.
    scene_composition = None
    try:
        import corpus_domain as _CD
        _card, _ = _CD.build_card_for_scene(card_id=out_dir.name, scene_type=scene_type, soil=host,
                                            objs=objs, coupling=None, note=meta["note"])
        (out_dir / "card.json").write_text(_card.to_json(), encoding="utf-8")
        scene_composition = _card.scene_composition.value
    except Exception as e:                                          # pragma: no cover
        log(f"  WARN domain card {out_dir.name}: {type(e).__name__}: {str(e)[:120]}")
    labels = {
        "scene_type": scene_type, "ambiguity": meta["ambiguity"], "note": meta["note"],
        "scene_composition": scene_composition,
        "host_material": meta["host_material"], "host_eps_r": meta["host_eps_r"],
        "host_coupling": host_coupling,
        "soil_heterogeneity": het,
        "objects": objs,
        "acquisition": {"fc_hz": meta["fc_hz"], "n_traces": meta["n_traces"], "dx_m": dx_m,
                        "dt_ns": dt_ns, "time_window_s": meta["time_window_s"],
                        "scan_start_m": round(scan_start, 4), "scan_step_m": round(scan_step, 5),
                        "scan_extent_m": [round(scan_start, 4), round(scan_start + (b.shape[1] - 1) * dx_m, 4)],
                        "scene_width_m": sc.width_m,
                        "scan_width_m": round(dx_m * b.shape[1], 4),
                        "scan_coverage": scan_cov},     # footprint preflight: complete-signature check
        "bscan_shape": list(b.shape), "material_order": r["material_order"],
    }
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    prov = {
        "engine": "gpr_agent.sim_scenes (build123d -> OCP voxelize -> gprMax HDF5)",
        "runtime": "gprMax GPU via subsurface_platform (GPR-Sim) build_gprmax_runtime_environment",
        "scene_type": scene_type, "seed": seed, "index": idx,
        "created_at": datetime.now(timezone.utc).isoformat(), "wall_s": round(time.time() - t0, 1),
        "scene": {"width_m": sc.width_m, "soil_depth_m": sc.soil_depth_m,
                  "dx_m": sc.dx_m, "soil_material": sc.soil_material},
        "meta": meta,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    _clean_caches(out_dir)
    with (OUT_ROOT / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"dir": str(out_dir.relative_to(OUT_ROOT)), "scene_type": scene_type,
                            "ambiguity": meta["ambiguity"], "n_objects": len(objs),
                            "scene_composition": scene_composition,
                            "host": meta["host_material"],
                            "host_realistic": drawn_realistic, "realistic_prob": realistic_host_prob,
                            "bscan_shape": list(b.shape),
                            "wall_s": prov["wall_s"], "created_at": prov["created_at"]}) + "\n")
    log(f"  OK  {scene_type:20s} {b.shape} host={meta['host_material']:14s}"
        f"[{'real ' if drawn_realistic else 'decorr'}] objs={len(objs)} {prov['wall_s']:.0f}s -> {out_dir.name}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=str, default="", help="comma list to restrict scene types")
    ap.add_argument("--n-traces", type=int, default=0, help="override traces/scene (verification)")
    # Decorrelation control. REQUIRED on purpose: the host-coupling strength is an experiment knob,
    # not a platform constant. p=1 -> always the type's plausible host (realistic, host leaks type);
    # p=0 -> host always random (fully decorrelated); 0<p<1 -> a measurable mix of both.
    ap.add_argument("--realistic-host-prob", type=float, required=True,
                    help="prob in [0,1] of drawing the type's geologically-plausible host; else a "
                         "random host (decorrelated). No default -- set it per run.")
    args = ap.parse_args()
    if not 0.0 <= args.realistic_host_prob <= 1.0:
        ap.error(f"--realistic-host-prob must be in [0,1]; got {args.realistic_host_prob}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logf = OUT_ROOT / "generation.log"

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with logf.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    _live = all_builders()                                  # natives + promoted (Tier T)
    types = [t for t in (args.only.split(",") if args.only else _live) if t in _live]
    pool = [t for t in types for _ in range(SCENE_WEIGHTS.get(t, 1))]
    rng = np.random.default_rng(args.seed)
    deadline = time.time() + args.hours * 3600.0
    log(f"=== corpus generation start: {len(types)} types, max={args.max}, "
        f"deadline={args.hours}h, realistic_host_prob={args.realistic_host_prob}, out={OUT_ROOT} ===")
    n_ok = n_fail = 0
    idx = 0
    while time.time() < deadline and n_ok < args.max:
        scene_type = str(rng.choice(pool))
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            run_one(scene_type, idx, seed, log, args.realistic_host_prob,
                    n_traces_override=args.n_traces)
            n_ok += 1
        except Exception as e:                                          # isolate every scene
            n_fail += 1
            log(f"  FAIL {scene_type}: {type(e).__name__}: {str(e)[:200]}")
            with (OUT_ROOT / "failures.log").open("a", encoding="utf-8") as f:
                f.write(f"\n=== {scene_type} idx={idx} seed={seed} {datetime.now().isoformat()} ===\n")
                f.write(traceback.format_exc())
        idx += 1
    log(f"=== done: {n_ok} ok, {n_fail} failed, {idx} attempts ===")


if __name__ == "__main__":
    main()

"""Decks-local / solve-remote corpus driver.

The proprietary stack stays LOCAL: we build the scene, voxelize it, and write the per-trace gprMax
decks here; only simulation DATA (geometry.h5 + materials.txt + t*.in, and the returned t*.out)
crosses the wire to a remote GPU. No stack code leaves the machine.

  1) python scripts/remote_corpus.py prepare  --n 6 --seed 1 --realistic-relation-prob 0.7 --out /tmp/prep
  2) <transfer /tmp/prep -> remote, run gprMax on each t*.in on the GPU, pull t*.out back>  (driven outside)
  3) python scripts/remote_corpus.py assemble --out /tmp/prep
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import corpus_domain as CD
import generate_composite_corpus as GC
import generate_subsurface_corpus as G
from gpr_agent.sim_scenes import _deck_single
from subsurface_platform.domain import geometry_in_domain, survey_in_domain   # GPR-Sim validators


def _obj_extents(o):
    """(x_min, x_max, z_min, z_max) bounding box of a labeled object (for geometry_in_domain)."""
    if "box_m" in o:
        b = o["box_m"]; return (b["x_min"], b["x_max"], b["depth_top"], b["depth_bottom"])
    if o.get("polygon_m"):
        xs = [p[0] for p in o["polygon_m"]]; zs = [p[1] for p in o["polygon_m"]]
        return (min(xs), max(xs), min(zs), max(zs))
    if "center_x_m" in o:
        cx, r, d = o["center_x_m"], o.get("radius_m", 0.0), o["depth_m"]
        return (cx - r, cx + r, max(0.0, d - r), d + r)
    return (0.0, 0.0, 0.0, 0.0)                              # non-localized -> degenerate (flagged)


def _acquisition(sc, objs, meta):
    """Reproduce run_composite_one's scan + footprint-coverage preflight; (scan_start, scan_step, n_traces, tw_s)."""
    scan_step = meta["scan_step_m"]; span = max(sc.width_m - 2 * G.SCAN_MARGIN_M, scan_step)
    n_traces = max(48, int(round(span / scan_step)) + 1)
    scan_start, tw_s = G.SCAN_MARGIN_M, meta["time_window_s"]
    cov = G._coverage_targets(objs)
    if cov:
        pre = G._coverage(cov, scan_x_min_m=scan_start, scan_x_max_m=scan_start + (n_traces - 1) * scan_step,
                          host_eps_r=meta["host_eps_r"], fc_hz=meta["fc_hz"], footprint_fn=G.footprint_half_width_m,
                          n_traces=n_traces, time_window_ns=tw_s * 1e9)
        if not pre["all_covered"]:
            scan_start = round(max(0.05, pre["recommended_scan_x_min_m"]), 4)
            scan_end = round(min(sc.width_m - 0.05, pre["recommended_scan_x_max_m"]), 4)
            n_traces = max(n_traces, int(round((scan_end - scan_start) / scan_step)) + 1,
                           int(pre.get("recommended_n_traces", n_traces)))
            scan_step = (scan_end - scan_start) / max(n_traces - 1, 1)
            tw_s = float(min(meta["tw_cap_s"], max(tw_s, pre["recommended_time_window_ns"] * 1e-9 * 1.1)))
    # DOMAIN preflight (GPR-Sim): keep every source + receiver inside the FDTD domain (the bug that
    # bit scene_00002). Auto-correct n_traces from the report, then re-check.
    dom_h = sc.soil_depth_m + sc.air_gap_m; ant_y = sc.soil_depth_m + 0.5 * sc.air_gap_m
    pf = dict(scan_x_min_m=scan_start, scan_step_m=scan_step, tx_rx_offset_m=meta["tx_rx_offset_m"],
              antenna_y_m=ant_y, domain_w_m=sc.width_m, domain_h_m=dom_h, edge_margin_m=0.02)
    rep = survey_in_domain(n_traces=n_traces, **pf)
    if not rep["all_inside"]:
        n_traces = rep["recommended_n_traces"]
        rep = survey_in_domain(n_traces=n_traces, **pf)
    return scan_start, scan_step, n_traces, tw_s, rep


# --------------------------------------------------------------------------------------------- #
# GSSI 400 MHz 3-D realism subset: the REAL antenna (antenna_like_GSSI_400) needs a 3-D domain at
# 2 mm, so these are bounded scenes (native #box/#cylinder/#sphere) -- ~26 s/trace on the A100.
# --------------------------------------------------------------------------------------------- #
GSSI_RES = 0.002
HOSTS_3D = {"dry_sand": (4.0, 0.001), "wet_clay": (22.0, 0.05), "loam": (12.0, 0.01),
            "moist_limestone": (10.0, 0.01), "silt": (12.0, 0.02)}
MAT_3D = {"pvc": (3.25, 0.0), "concrete": (6.5, 0.005), "water": (81.0, 0.05)}


def _gssi_scene(rng, rrp):
    """A bounded 3-D scene for the GSSI-400 antenna (native commands). Returns a prep-style dict."""
    host = str(rng.choice(list(HOSTS_3D))); Dy = 0.40
    # keep the 3-D grid small (2 mm forces a fine mesh): ~1 m x 0.4 m x ~1.1 m -> ~6e7 cells, ~40 s/trace
    Dx = float(rng.uniform(0.9, 1.3)); depth = float(rng.uniform(0.8, 1.2))
    surf_z = round(depth, 3); Dz = round(surf_z + 0.30, 3)              # ground 0..surf_z, air above for the antenna
    realistic = bool(rng.random() < rrp)
    mats = {host: HOSTS_3D[host]}
    cmds = [f"#box: 0 0 0 {Dx:.3f} {Dy:.3f} {surf_z:.3f} {host}"]
    objs = []
    tcx = None
    if rng.random() < 0.5:                                             # a trench backfill box (fits the small domain)
        bw = float(rng.uniform(0.3, max(0.32, Dx - 0.5))); tcx = float(rng.uniform(bw / 2 + 0.12, Dx - bw / 2 - 0.12))
        tbot = float(rng.uniform(0.5, min(1.0, depth - 0.2))); bf = str(rng.choice(["dry_sand", "silt"]))
        mats[bf] = HOSTS_3D[bf]
        cmds.append(f"#box: {tcx-bw/2:.3f} 0 {surf_z-tbot:.3f} {tcx+bw/2:.3f} {Dy:.3f} {surf_z:.3f} {bf}")
        objs.append({"kind": "trench_backfill", "material": bf, "conductor": False,
                     "center_x_m": round(tcx, 3), "depth_m": round(tbot / 2, 3), "radius_m": round(bw / 2, 3)})
    for _ in range(int(rng.integers(1, 4))):                           # 1-3 pipes (varied type)
        if realistic and tcx is not None:
            px = float(np.clip(rng.uniform(tcx - 0.25, tcx + 0.25), 0.3, Dx - 0.3)); pzd = float(rng.uniform(0.3, depth - 0.2))
        else:
            px = float(rng.uniform(0.3, Dx - 0.3)); pzd = float(rng.uniform(0.2, depth - 0.15))
        pz = surf_z - pzd; r = float(rng.uniform(0.03, 0.06))
        code = str(rng.choice(["metal", "pvc_empty", "pvc_water", "concrete"]))
        if code == "metal":
            cmds.append(f"#cylinder: {px:.3f} 0 {pz:.3f} {px:.3f} {Dy:.3f} {pz:.3f} {r:.3f} pec")
            objs.append({"kind": "pipe", "material": "steel", "ptype": "metal", "fill": "none", "conductor": True,
                         "center_x_m": round(px, 3), "depth_m": round(pzd, 3), "radius_m": round(r, 3)})
        elif code == "concrete":
            mats["concrete"] = MAT_3D["concrete"]
            cmds.append(f"#cylinder: {px:.3f} 0 {pz:.3f} {px:.3f} {Dy:.3f} {pz:.3f} {r:.3f} concrete")
            objs.append({"kind": "pipe", "material": "concrete", "ptype": "concrete", "fill": "none", "conductor": False,
                         "center_x_m": round(px, 3), "depth_m": round(pzd, 3), "radius_m": round(r, 3)})
        else:
            mats["pvc"] = MAT_3D["pvc"]; water = code == "pvc_water"
            cmds.append(f"#cylinder: {px:.3f} 0 {pz:.3f} {px:.3f} {Dy:.3f} {pz:.3f} {r:.3f} pvc")
            fill = "water" if water else "free_space"
            if water:
                mats["water"] = MAT_3D["water"]
            cmds.append(f"#cylinder: {px:.3f} 0 {pz:.3f} {px:.3f} {Dy:.3f} {pz:.3f} {r*0.6:.3f} {fill}")
            objs.append({"kind": "pipe", "material": "pvc", "fill": "water" if water else "air", "conductor": False,
                         "ptype": "water_filled_plastic" if water else "empty_plastic",
                         "center_x_m": round(px, 3), "depth_m": round(pzd, 3), "radius_m": round(r, 3)})
    if rng.random() < 0.4:                                             # a spherical air void
        vx = float(rng.uniform(0.3, Dx - 0.3)); vzd = float(rng.uniform(0.3, depth - 0.2)); vr = float(rng.uniform(0.06, 0.12))
        cmds.append(f"#sphere: {vx:.3f} {Dy/2:.3f} {surf_z-vzd:.3f} {vr:.3f} free_space")
        objs.append({"kind": "karst_void", "material": "air", "shape": "blob", "cavity_type": "karst_void",
                     "conductor": False, "center_x_m": round(vx, 3), "depth_m": round(vzd, 3), "radius_m": round(vr, 3)})
    v = 0.3 / max(HOSTS_3D[host][0], 1.0) ** 0.5
    tw = float(min(4.5e-8, (2.2 * depth / v + 6.0) * 1e-9))
    x0, x1, step = 0.18, Dx - 0.18, 0.04
    n_traces = max(10, int((x1 - x0) / step) + 1)
    mat_lines = [f"#material: {e:.4g} {s:.5g} 1 0 {nm}" for nm, (e, s) in mats.items()]
    meta = {"tier": "M", "scale": "gssi3d", "family": "gssi_400", "coupling": "ground_coupled",
            "antenna": "gssi_400", "fc_hz": 4.0e8, "dx_m": GSSI_RES, "tx_rx_offset_m": 0.0,
            "host_eps_r": round(HOSTS_3D[host][0], 3),
            "note": f"gssi_400 3D: {host}, {len([o for o in objs if o['kind']=='pipe'])} pipes",
            "correlation_sampled": [], "scan_step_m": step}
    return {"Dx": Dx, "Dy": Dy, "Dz": Dz, "surf_z": surf_z, "mat_lines": mat_lines, "cmds": cmds,
            "objs": objs, "host": host, "x0": x0, "step": step, "n_traces": n_traces, "tw": tw,
            "realistic": realistic, "meta": meta}


def _gssi_tu1208_scene(rng, rrp):
    """A TU1208-like 3-D scene for the GSSI-400 antenna: a pit + THREE depth layers of pipes BY TYPE
    (steel shallow / water-PVC mid / empty-PVC deep -- the TU1208 layering) + an optional polystyrene
    cavity + an optional gneiss block, in a limestone host (eps~6). Feasible GSSI domain (~1.6x1.35x0.4 m).
    Returns the same prep-style dict as _gssi_scene."""
    LIMESTONE = (6.0, 0.005)                                          # TU1208 host eps ~ 6
    Dy = 0.40
    Dx = float(rng.uniform(1.4, 1.8)); depth = float(rng.uniform(1.15, 1.5))
    surf_z = round(depth, 3); Dz = round(surf_z + 0.30, 3)
    realistic = bool(rng.random() < rrp)
    mats = {"limestone": LIMESTONE}
    cmds = [f"#box: 0 0 0 {Dx:.3f} {Dy:.3f} {surf_z:.3f} limestone"]
    objs = []
    pcx = Dx / 2; top_w = float(rng.uniform(Dx * 0.72, Dx * 0.92))    # the pit (backfill box ~ trapezoidal)
    pit_d = float(rng.uniform(0.85, min(1.1, depth - 0.2)))
    bf = str(rng.choice(["dry_sand", "silt"])); mats[bf] = HOSTS_3D[bf]
    cmds.append(f"#box: {pcx-top_w/2:.3f} 0 {surf_z-pit_d:.3f} {pcx+top_w/2:.3f} {Dy:.3f} {surf_z:.3f} {bf}")
    objs.append({"kind": "trench_backfill", "material": bf, "conductor": False, "center_x_m": round(pcx, 3),
                 "depth_m": round(pit_d / 2, 3), "radius_m": round(top_w / 2, 3)})
    mats["pvc"] = MAT_3D["pvc"]; mats["water"] = MAT_3D["water"]
    for code, base in [("metal", rng.uniform(0.30, 0.42)), ("pvc_water", rng.uniform(0.58, 0.74)),
                       ("pvc_empty", rng.uniform(0.90, 1.05))]:        # 3 TU1208 layers, 1-3 pipes each
        pzd = float(min(base, depth - 0.12))
        for x in np.linspace(pcx - top_w / 2 + 0.18, pcx + top_w / 2 - 0.18, int(rng.integers(1, 4))):
            px = float(np.clip(x + rng.uniform(-0.04, 0.04), 0.25, Dx - 0.25)); pz = surf_z - pzd
            r = float(rng.uniform(0.03, 0.06))
            if code == "metal":
                cmds.append(f"#cylinder: {px:.3f} 0 {pz:.3f} {px:.3f} {Dy:.3f} {pz:.3f} {r:.3f} pec")
                objs.append({"kind": "pipe", "material": "steel", "ptype": "metal", "fill": "none", "conductor": True,
                             "center_x_m": round(px, 3), "depth_m": round(pzd, 3), "radius_m": round(r, 3), "tu1208_layer": code})
            else:
                water = code == "pvc_water"
                cmds.append(f"#cylinder: {px:.3f} 0 {pz:.3f} {px:.3f} {Dy:.3f} {pz:.3f} {r:.3f} pvc")
                cmds.append(f"#cylinder: {px:.3f} 0 {pz:.3f} {px:.3f} {Dy:.3f} {pz:.3f} {r*0.6:.3f} {'water' if water else 'free_space'}")
                objs.append({"kind": "pipe", "material": "pvc", "fill": "water" if water else "air",
                             "ptype": "water_filled_plastic" if water else "empty_plastic", "conductor": False,
                             "center_x_m": round(px, 3), "depth_m": round(pzd, 3), "radius_m": round(r, 3), "tu1208_layer": code})
    if rng.random() < 0.55:                                            # polystyrene cavity (TU1208 has one)
        vx = float(rng.uniform(0.3, Dx - 0.3)); vzd = float(rng.uniform(0.4, depth - 0.2)); vr = float(rng.uniform(0.07, 0.13))
        cmds.append(f"#sphere: {vx:.3f} {Dy/2:.3f} {surf_z-vzd:.3f} {vr:.3f} free_space")
        objs.append({"kind": "polystyrene_cavity", "material": "air", "shape": "sphere", "cavity_type": "polystyrene_cavity",
                     "conductor": False, "center_x_m": round(vx, 3), "depth_m": round(vzd, 3), "radius_m": round(vr, 3)})
    if rng.random() < 0.4:                                             # gneiss block (TU1208 blocks)
        mats["gneiss"] = (9.0, 0.005)
        gx = float(rng.uniform(0.3, Dx - 0.3)); gzd = float(rng.uniform(0.55, depth - 0.2)); gw = float(rng.uniform(0.10, 0.20))
        cmds.append(f"#box: {gx-gw/2:.3f} {Dy/2-gw/2:.3f} {surf_z-gzd-gw/2:.3f} {gx+gw/2:.3f} {Dy/2+gw/2:.3f} {surf_z-gzd+gw/2:.3f} gneiss")
        objs.append({"kind": "gneiss_block", "material": "gneiss", "conductor": False,
                     "center_x_m": round(gx, 3), "depth_m": round(gzd, 3), "radius_m": round(gw / 2, 3)})
    v = 0.3 / max(LIMESTONE[0], 1.0) ** 0.5
    tw = float(min(5.0e-8, (2.2 * depth / v + 6.0) * 1e-9))
    x0, x1, step = 0.20, Dx - 0.20, 0.04
    n_traces = max(10, int((x1 - x0) / step) + 1)
    mat_lines = [f"#material: {e:.4g} {sg:.5g} 1 0 {nm}" for nm, (e, sg) in mats.items()]
    meta = {"tier": "M", "scale": "gssi3d", "family": "gssi_400_tu1208", "coupling": "ground_coupled",
            "antenna": "gssi_400", "fc_hz": 4.0e8, "dx_m": GSSI_RES, "tx_rx_offset_m": 0.0,
            "host_eps_r": round(LIMESTONE[0], 3),
            "note": "gssi_400 TU1208-like: limestone, 3 pipe layers (steel/water-PVC/empty-PVC)",
            "correlation_sampled": [], "scan_step_m": step}
    return {"Dx": Dx, "Dy": Dy, "Dz": Dz, "surf_z": surf_z, "mat_lines": mat_lines, "cmds": cmds,
            "objs": objs, "host": "limestone", "x0": x0, "step": step, "n_traces": n_traces, "tw": tw,
            "realistic": realistic, "meta": meta}


def _gssi_deck(s, sx):
    """One GSSI-400 trace deck: native geometry + the real antenna (#python) stepped to source x = sx."""
    return "\n".join([
        "#title: GSSI 400MHz 3-D trace",
        f"#domain: {s['Dx']:.3f} {s['Dy']:.3f} {s['Dz']:.3f}",
        f"#dx_dy_dz: {GSSI_RES} {GSSI_RES} {GSSI_RES}",
        f"#time_window: {s['tw']:.3g}", "",
        *s["mat_lines"], *s["cmds"], "",
        "#python:",
        "import os, sys",
        "sys.path.insert(0, os.path.expanduser('~/gprMax'))",
        "from user_libs.antennas.GSSI import antenna_like_GSSI_400",
        f"antenna_like_GSSI_400({sx:.3f}, {s['Dy']/2:.3f}, {s['surf_z']:.3f}, resolution={GSSI_RES})",
        "#end_python:", "",
    ]) + "\n"


def _prepare_one(idx, s, rrp, gssi_frac, out, tu1208_gssi=False):
    """Prepare one scene (decks + prep.json). Returns True if a scene was written, False if skipped."""
    srng = np.random.default_rng(s)
    if tu1208_gssi or srng.random() < gssi_frac:              # GSSI-400 3-D realism subset (real antenna)
        gs = _gssi_tu1208_scene(srng, rrp) if tu1208_gssi else _gssi_scene(srng, rrp)
        gr = geometry_in_domain([_obj_extents(o) for o in gs["objs"]], width_m=gs["Dx"], depth_m=gs["surf_z"])
        if not gr["all_inside"]:                              # validator gate: no out-of-domain geometry
            print(f"  GEOM FAIL {idx} (gssi): {gr['issues'][:2]}"); return False
        d = out / f"scene_{idx:05d}"; d.mkdir(exist_ok=True)
        for k in range(gs["n_traces"]):
            (d / f"t{k:03d}.in").write_text(_gssi_deck(gs, gs["x0"] + k * gs["step"]), encoding="utf-8")
        (d / "prep.json").write_text(json.dumps(
            {"n_traces": gs["n_traces"], "scan_start": gs["x0"], "scan_step": gs["step"], "fc_hz": 4.0e8,
             "tw_s": gs["tw"], "host": gs["host"], "width": gs["Dx"], "depth": gs["surf_z"], "meta": gs["meta"],
             "objects": gs["objs"], "material_order": [], "scene_composition": "composite", "n_relations": None,
             "realistic_relation": gs["realistic"], "component": "Ey", "preflight": {"all_inside": True}},
            default=str, indent=2), encoding="utf-8")
        print(f"PREP {idx:03d} gssi_400    3D         host={gs['host']:14s} traces={gs['n_traces']:3d} -> {d.name}")
        return True
    host_rng = np.random.default_rng([s, 0x484F5354])
    host = str(host_rng.choice(G.SOILS)); rr = bool(host_rng.random() < rrp)
    sc, objs, meta = GC.build_composite_corridor(srng, host_soil=host, realistic_relation=rr)
    ctx = {"pipe_void", "pipe_fill_water", "trench_backfill"}
    tk = {o["kind"] for o in objs if o["kind"] not in ctx and not o["kind"].endswith(("_layer", "_stratum"))}
    if tk <= {"pipe"}:
        print(f"  skip {idx}: pipe-only"); return False
    gr = geometry_in_domain([_obj_extents(o) for o in objs], width_m=sc.width_m, depth_m=sc.soil_depth_m)
    if not gr["all_inside"]:                                  # validator gate: no out-of-domain geometry
        print(f"  GEOM FAIL {idx}: {gr['issues'][:2]}"); return False
    sc.soil_material = host; meta["host_material"] = host; meta["host_eps_r"] = round(G._eps(host), 3)
    sc.soil_heterogeneity = {"eps_spread_frac": 0.10, "correlation_length_m": float(srng.uniform(0.1, 0.25)),
                             "n_levels": 9, "seed": s}
    scan_start, scan_step, n_traces, tw_s, pf = _acquisition(sc, objs, meta)
    if not pf["all_inside"]:                                  # preflight gate: never ship an invalid deck
        print(f"  PREFLIGHT FAIL {idx}: {pf['issues']}"); return False
    d = out / f"scene_{idx:05d}"; d.mkdir(exist_ok=True)
    geo, mats, order = sc.write_gprmax(d)
    for k in range(n_traces):
        sx = scan_start + k * scan_step
        deck = _deck_single(sc, Path("geometry.h5"), Path("materials.txt"), fc_hz=meta["fc_hz"],
                            source_x=sx, tx_rx_offset_m=meta["tx_rx_offset_m"], time_window_s=tw_s)
        (d / f"t{k:03d}.in").write_text(deck, encoding="utf-8")
    try:
        GC._render_model_png(d, meta, host)
    except Exception:
        pass
    comp = rels = None
    try:
        card, _ = CD.build_card_for_scene(card_id=d.name, scene_type=GC.SCENE_TYPE, soil=host,
                                          objs=objs, coupling=None, note=meta["note"])
        comp = card.scene_composition.value; rels = len(card.scene_relations)
        (d / "card.json").write_text(card.to_json(), encoding="utf-8")
    except Exception:
        pass
    (d / "prep.json").write_text(json.dumps(
        {"n_traces": n_traces, "scan_start": scan_start, "scan_step": scan_step, "fc_hz": meta["fc_hz"],
         "tw_s": tw_s, "host": host, "width": sc.width_m, "depth": sc.soil_depth_m, "meta": meta,
         "objects": objs, "material_order": order, "scene_composition": comp, "n_relations": rels,
         "realistic_relation": rr, "preflight": pf, "component": "Ez"}, default=str, indent=2), encoding="utf-8")
    print(f"PREP {idx:03d} {meta.get('family','?'):10s} {meta['scale']:10s} host={host:14s} "
          f"traces={n_traces:3d} -> {d.name}")
    return True


def prepare(n, seed, rhp, rrp, out, gssi_frac=0.3, tu1208_gssi=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); made = 0
    for idx in range(n):
        s = int(rng.integers(0, 2**31 - 1))
        try:                                                  # isolate every scene -- one failure can't abort the run
            if _prepare_one(idx, s, rrp, gssi_frac, out, tu1208_gssi=tu1208_gssi):
                made += 1
        except Exception as e:
            print(f"  FAIL idx={idx}: {type(e).__name__}: {str(e)[:160]}")
    print(f"PREPARE_DONE made={made} out={out}")


def _assemble_one(d, OUT, h5py):
    """Assemble one prepared scene from its t*.out into the corpus. Returns 1 if done, 0 if skipped."""
    pj = json.loads((d / "prep.json").read_text())
    nt = pj["n_traces"]; sol = {}; dt = None; comp = pj.get("component", "Ez")   # GSSI-400 outputs Ey
    for k in range(nt):
        o = d / f"t{k:03d}.out"
        if not o.exists():
            continue
        with h5py.File(o, "r") as f:
            dt = float(f.attrs["dt"]); sol[k] = np.asarray(f["rxs"]["rx1"][comp], float)
    if len(sol) < 0.9 * nt:                                       # need most traces; else not worth assembling
        print(f"  skip {d.name}: only {len(sol)}/{nt} traces solved"); return 0
    ns = len(next(iter(sol.values()))); b = np.zeros((ns, nt))
    for k, tr in sol.items():
        b[:, k] = tr
    if len(sol) < nt:                                             # pad the few missing with a neighbour
        for k in range(nt):
            if k not in sol:
                b[:, k] = b[:, k - 1] if k > 0 else b[:, k + 1]
        print(f"  {d.name}: padded {nt - len(sol)} missing trace(s)")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    sd = OUT / f"composite_{ts}_{d.name[-5:]}"; sd.mkdir(parents=True, exist_ok=True)
    np.save(sd / "bscan.npy", b)
    for fn in ("geometry.h5", "materials.txt", "model.png", "card.json"):
        if (d / fn).exists():
            (sd / fn).write_bytes((d / fn).read_bytes())
    G._save_preview(sd / "bscan.png", b)
    meta = pj["meta"]; dx = meta["dx_m"]
    labels = {"scene_type": GC.SCENE_TYPE, "tier": meta["tier"], "scale": meta["scale"], "family": meta.get("family"),
              "coupling": meta["coupling"], "antenna": meta["antenna"], "host_material": pj["host"],
              "host_eps_r": meta.get("host_eps_r"), "scene_composition": pj["scene_composition"],
              "realistic_relation": pj["realistic_relation"], "correlation_sampled": meta.get("correlation_sampled", []),
              "objects": pj["objects"], "note": meta["note"],
              "acquisition": {"fc_hz": meta["fc_hz"], "n_traces": nt, "dx_m": dx, "dt_ns": dt * 1e9,
                              "time_window_s": pj["tw_s"], "scan_start_m": pj["scan_start"],
                              "scan_step_m": pj["scan_step"], "scene_width_m": pj["width"], "depth_m": pj["depth"]},
              "bscan_shape": list(b.shape), "material_order": pj["material_order"], "solve": "remote_gpu_a100"}
    (sd / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    with (G.OUT_ROOT / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"dir": str(sd.relative_to(G.OUT_ROOT)), "scene_type": GC.SCENE_TYPE, "tier": meta["tier"],
                            "scale": meta["scale"], "family": meta.get("family"), "coupling": meta["coupling"],
                            "n_objects": len(pj["objects"]), "scene_composition": pj["scene_composition"],
                            "host": pj["host"], "realistic_relation": pj["realistic_relation"],
                            "bscan_shape": list(b.shape), "solve": "remote_gpu_a100", "created_at": ts}) + "\n")
    print(f"ASSEMBLE {d.name} -> {sd.name}  bscan {b.shape}")
    return 1


def assemble(out):
    import h5py
    out = Path(out); OUT = G.OUT_ROOT / GC.SCENE_TYPE; OUT.mkdir(parents=True, exist_ok=True); done = 0
    for d in sorted(out.glob("scene_*")):
        try:
            done += _assemble_one(d, OUT, h5py)
        except Exception as e:
            print(f"  ASSEMBLE FAIL {d.name}: {type(e).__name__}: {str(e)[:140]}")
    print(f"ASSEMBLE_DONE n={done}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "assemble"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realistic-host-prob", type=float, default=1.0)
    ap.add_argument("--realistic-relation-prob", type=float, default=0.7)
    ap.add_argument("--gssi-frac", type=float, default=0.3, help="fraction of scenes using the GSSI-400 3-D antenna")
    ap.add_argument("--gssi-tu1208", action="store_true", help="TU1208-like GSSI scenes (3 pipe layers + pit + cavity)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare(a.n, a.seed, a.realistic_host_prob, a.realistic_relation_prob, a.out,
                gssi_frac=a.gssi_frac, tu1208_gssi=a.gssi_tu1208)
    else:
        assemble(a.out)


if __name__ == "__main__":
    main()

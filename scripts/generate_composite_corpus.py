"""Tier-L composite corridor corpus: large (TU1208-class) multi-object scenes whose object
PRESENCE + ARRANGEMENT come from the object<->object correlation table.

Reuses the proven Tier-S generator (`generate_subsurface_corpus`) for the scene API + helpers,
and the object<->object co-occurrence/arrangement rules
(`subsurface_platform.domain.object_correlation`). A corridor = a primary utility trench (with
bedded pipes), a parallel-conduit duct bank, an optional deep crossing service, and host clutter,
laid out across ~3 depth bands at 8-20 m width / 3-4.5 m depth, imaged at 250 MHz on GPU.

Both correlation strengths are operator inputs (mirroring the host policy):
  --realistic-host-prob      object<->host presence/shape/material/depth (Tier-S knob, reused)
  --realistic-relation-prob  object<->object presence + arrangement (this script's knob)
At p<1 the corresponding correlation is broken (off-distribution), tagged in the manifest, so a
detector cannot exploit "a trench usually contains a pipe" / "a duct bank's conduits are parallel".

Run (PYTHONPATH = GPR-Agent/src;GPR-Agent/scripts;GPR-Sim/src;GPR-KnowledgeBase;GPR-Tools/src;
GPR-Interpretation/src; CUDA v12.8 bin on PATH for the GPU):
    python scripts/generate_composite_corpus.py --n 40 --seed 7 \
        --realistic-host-prob 1.0 --realistic-relation-prob 0.7
    python scripts/generate_composite_corpus.py --dry --n 5 ...      # build only, no gprMax
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from datetime import datetime, timezone

import numpy as np

import corpus_domain as CD
import generate_subsurface_corpus as G          # scene API + helpers (_obj/_meta/_eps/_coverage/...)
from subsurface_platform.domain.host_correlation import correlated_shape_for
from subsurface_platform.domain.object_correlation import (
    CLUTTER_COMPANIONS, Arrangement, sample_companions)
from subsurface_platform.domain.scenario_relation import ScenarioRelationType

SCENE_TYPE = "composite_corridor"
SCAN_STEP_L_M = 0.05            # coarser line spacing for 250 MHz (anti-alias-checked below)
DX_L_M = 0.01                  # 10 mm voxel (TU1208 scale)
TW_CAP_L_S = 1.5e-7            # up to 150 ns (deep scene)


GEO_STRATA = ("topsoil_moist", "loam", "wet_clay", "dry_sand", "gravel", "silt")
PAVEMENT = (("asphalt", 0.06, 0.12), ("gravel", 0.15, 0.30), ("dry_sand", 0.15, 0.35))   # surface->down, over subgrade


def _add_layers(sc, rng, kind, D, W):
    """Add a horizontally LAYERED ground (pavement or geological strata) before any pit. Layers are
    full-width; a later pit paints over them (cuts through). Returns (layer objs, note)."""
    objs = []; names = []; z = 0.0
    if kind == "pavement":
        for mat, lo, hi in PAVEMENT:
            th = float(rng.uniform(lo, hi))
            if z + th > D - 0.5:
                break
            sc.add_layer(depth_top_m=z, thickness_m=th, material=mat)
            objs.append(G._obj(f"{mat}_layer", mat, box={"x_min": 0.0, "x_max": round(W, 3),
                                                         "depth_top": round(z, 3), "depth_bottom": round(z + th, 3)}))
            names.append(mat); z += th
        return objs, "pavement(" + "/".join(names) + ")"
    # geological strata are NOT flat: build each as a polygon band whose top/bottom interfaces follow a
    # shared low-frequency wave + an overall dip (the air-ground surface stays flat).
    strata = list(rng.choice(GEO_STRATA, size=int(rng.integers(3, 5)), replace=False))
    n = len(strata); nx = 28; xs = np.linspace(0.0, W, nx)
    dip = float(rng.uniform(-0.12, 0.12)); amp = float(rng.uniform(0.08, 0.22))
    f1, f2 = float(rng.uniform(0.6, 1.6)), float(rng.uniform(1.6, 3.2))
    ph1, ph2 = float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 2 * np.pi))
    wave = (amp * (0.7 * np.sin(2 * np.pi * f1 * xs / W + ph1) + 0.3 * np.sin(2 * np.pi * f2 * xs / W + ph2))
            + dip * D * (xs / W - 0.5))
    base = [0.0]
    for i in range(n):
        th = float(rng.uniform(0.35, max(0.5, (D - z) / max(n - i, 1)))); z = min(z + th, D - 0.2); base.append(z)
    iface = [np.zeros(nx)] + [np.clip(base[i] + wave * float(rng.uniform(0.6, 1.1)), 0.05, D) for i in range(1, n + 1)]
    for i in range(1, n + 1):                                  # keep interfaces from crossing
        iface[i] = np.maximum(iface[i], iface[i - 1] + 0.05)
    for i in range(n):
        top, bot = iface[i], iface[i + 1]
        poly = ([(float(xs[j]), float(top[j])) for j in range(nx)]
                + [(float(xs[nx - 1 - j]), float(bot[nx - 1 - j])) for j in range(nx)])
        sc.add_polygon(corners_xz=poly, material=str(strata[i]), name=f"{strata[i]}_stratum")
        objs.append(G._obj(f"{strata[i]}_stratum", str(strata[i]),
                           box={"x_min": 0.0, "x_max": round(W, 3),
                                "depth_top": round(float(top.mean()), 3), "depth_bottom": round(float(bot.mean()), 3)}))
        names.append(str(strata[i]))
    return objs, "strata~(" + "/".join(names) + ")"


def _add_cavity(sc, rng, objs, cx, cz, r, material, kind, *, shape=None):
    """Add a cavity/void with a varied SHAPE (consumes the placement-grammar polygon path):
      circle  -- simple air pocket / pipe void
      ellipse -- a flattened lens / perched air pocket
      dome    -- an arched roof over a flat floor (the TU1208 polystyrene; a collapse/tunnel crown)
      rect    -- a buried box: utility vault, culvert, chamber, basement
      chimney -- a raveling sinkhole void (narrow neck, bulbous body; either orientation)
      blob    -- an irregular karst dissolution void
    """
    shape = shape or str(rng.choice(["circle", "ellipse", "dome", "rect", "chimney", "blob", "blob"]))
    poly = None; eff = r
    if shape == "circle":
        sc.add_void(center_x_m=cx, depth_m=cz, radius_m=r, material=material)
    elif shape == "ellipse":
        a = r * float(rng.uniform(1.1, 1.9)); b = r * float(rng.uniform(0.45, 0.85)); rot = float(rng.uniform(0, np.pi))
        t = np.linspace(0, 2 * np.pi, 20, endpoint=False); ex, ey = a * np.cos(t), b * np.sin(t)
        poly = [(cx + ex[i] * np.cos(rot) - ey[i] * np.sin(rot), cz + ex[i] * np.sin(rot) + ey[i] * np.cos(rot)) for i in range(len(t))]
        eff = a
    elif shape == "dome":                                     # arched roof, flat floor (TU1208 polystyrene)
        w = r * float(rng.uniform(1.0, 1.6)); h = r * float(rng.uniform(0.9, 1.5)); zf = cz + h / 2
        t = np.linspace(0.0, np.pi, 16)
        poly = [(cx - w, zf), (cx + w, zf)] + [(cx + w * np.cos(tt), zf - h * np.sin(tt)) for tt in t]; eff = max(w, h)
    elif shape == "rect":                                     # buried box / vault / culvert
        w = r * float(rng.uniform(1.3, 2.4)); h = r * float(rng.uniform(0.6, 1.5))
        poly = [(cx - w / 2, cz - h / 2), (cx + w / 2, cz - h / 2), (cx + w / 2, cz + h / 2), (cx - w / 2, cz + h / 2)]; eff = max(w, h) / 2
    elif shape == "chimney":                                  # raveling sinkhole: narrow neck + bulb
        wt = r * float(rng.uniform(0.3, 0.6)); wb = r * float(rng.uniform(1.0, 1.6)); h = r * float(rng.uniform(1.3, 2.2))
        if rng.random() < 0.5:
            wt, wb = wb, wt
        poly = [(cx - wt, cz - h / 2), (cx + wt, cz - h / 2), (cx + wb, cz + h / 2), (cx - wb, cz + h / 2)]; eff = max(wb, h / 2)
    else:                                                     # irregular karst blob
        m = int(rng.integers(7, 12)); ang = np.sort(rng.uniform(0, 2 * np.pi, m)); rr = r * (1 + rng.uniform(-0.4, 0.6, m))
        poly = [(cx + rr[i] * np.cos(ang[i]), cz + rr[i] * np.sin(ang[i])) for i in range(m)]; eff = float(rr.max())
    # cavity TYPE is semantically tied to its shape -- record both so cavities are separable by type.
    shape_type = {"circle": "air_pocket", "ellipse": "air_lens", "dome": "dome_void",
                  "rect": "utility_vault", "chimney": "sinkhole_void", "blob": "karst_void"}
    ctype = shape_type[shape]
    label = kind if (kind and kind != "generic_void") else ctype     # keep forced kinds (e.g. polystyrene_cavity)
    if poly is not None:
        poly = [(float(px), float(max(0.05, pz))) for px, pz in poly]
        sc.add_polygon(corners_xz=poly, material=material, name=label)
    o = G._obj(label, material, x=cx, depth=cz, radius=float(eff), polygon=poly, ambiguity=True)
    o["shape"] = shape; o["cavity_type"] = ctype
    objs.append(o)


def build_composite_corridor(rng, host_soil=None, realistic_relation=True):
    """A composite subsurface scene in one of four FAMILIES:
      excavation -- a wide backfilled pit with utilities layered inside + host clutter.
      tu1208     -- the IFSTTAR TU1208 object set: a trapezoidal limestone pit in rock, 3 pipe layers
                    (steel / water-filled PVC / empty PVC), gneiss blocks, a polystyrene cavity.
      geological -- horizontally layered soil strata cut by a utility trench + utilities.
      pavement   -- an asphalt/base/subbase pavement cut by a utility trench + utilities.
    Object presence + the layered-vs-scattered arrangement come from object_correlation (strength-gated);
    pipe locations + counts vary per scene."""
    family = str(rng.choice(["excavation", "tu1208", "geological", "pavement"], p=[0.45, 0.25, 0.18, 0.12]))
    is_tu, is_pav, is_geo = family == "tu1208", family == "pavement", family == "geological"

    if is_tu:                                                      # TU1208: rock host, large trapezoidal pit
        soil = host_soil or str(rng.choice(["granite", "dry_limestone", "moist_limestone"]))
        large = True; W = float(rng.uniform(12.0, 20.0)); D = float(rng.uniform(3.2, 4.2)); dx = round(float(rng.uniform(0.008, 0.014)), 4)   # vary dx (=> dt, n_samples vary)
        pit_frac = float(rng.uniform(0.6, 0.85))
    elif is_pav:                                                  # pavement: shallow, over a sandy subgrade
        soil = host_soil or "dry_sand"
        large = False; W = float(rng.uniform(4.0, 8.0)); D = float(rng.uniform(1.6, 2.4)); dx = round(float(rng.uniform(0.004, 0.008)), 4)   # vary dx (=> dt, n_samples vary)
        pit_frac = float(rng.uniform(0.25, 0.5))
    else:                                                          # excavation / geological -- scale-matched canvas
        large = bool(rng.random() < 0.45)
        soil = host_soil or str(rng.choice(G.SOILS))
        if large:
            W = float(rng.uniform(10.0, 18.0)); D = float(rng.uniform(3.0, 4.0)); dx = round(float(rng.uniform(0.008, 0.014)), 4)   # vary dx (=> dt, n_samples vary)
            pit_frac = float(rng.uniform(0.5, 0.78))
        else:
            W = float(rng.uniform(3.5, 6.5)); D = float(rng.uniform(1.8, 2.8)); dx = round(float(rng.uniform(0.004, 0.008)), 4)   # vary dx (=> dt, n_samples vary)
            pit_frac = float(rng.uniform(0.3, 0.6))
    sc = G.S.Scene(width_m=W, soil_depth_m=D, dx_m=dx, soil_material=soil)
    # COUPLING: the antenna sits in the air gap -> air_gap_m sets ground-coupled vs air-launched.
    air_launched = bool(rng.random() < (0.6 if is_pav else 0.3))   # road surveys are often air-launched
    sc.air_gap_m = float(rng.uniform(0.25, 0.5)) if air_launched else float(rng.uniform(0.02, 0.06))
    coupling = "air_launched" if air_launched else "ground_coupled"
    # ANTENNA: 2-D hertzian dipole; vary fc by scale + the Tx-Rx offset (3-D MALA is the separate path).
    fc = 2.5e8 if large else float(rng.choice([5.0e8, 8.0e8])); tx_rx_offset = round(float(rng.uniform(0.04, 0.18)), 3)
    objs: list[dict] = []
    notes: list[str] = []
    sampled: list[dict] = []

    def _record(anchor, companion, relation, arr):
        sampled.append({"anchor": anchor, "companion": companion,
                        "relation": relation.value, "arrangement": arr.value})

    # --- layered ground (host stratification), added FIRST so the pit cuts through it ---
    if is_pav or is_geo:
        lobjs, lnote = _add_layers(sc, rng, "pavement" if is_pav else "geological", D, W)
        objs += lobjs; notes.append(lnote)

    # --- backfilled pit / trench (cuts through any layers); shape host-correlated (tu1208 = trapezoid) ---
    backfill = "moist_limestone" if is_tu else str(rng.choice(["dry_sand", "gravel", "silt", "moist_limestone"]))
    pit_w = pit_frac * W
    pit_cx = float(np.clip(W / 2 + rng.uniform(-0.08, 0.08) * W, pit_w / 2 + 0.3, W - pit_w / 2 - 0.3))
    pit_bot = float(rng.uniform(min(1.2, D - 0.6), D - 0.3)); htop = pit_w / 2
    shape = "trapezoid" if is_tu else correlated_shape_for("utility_trench", soil, rng, realistic=realistic_relation)
    if shape == "box":
        hbot = htop
        sc.add_box(x_min_m=pit_cx - htop, x_max_m=pit_cx + htop, depth_top_m=0.0, depth_bottom_m=pit_bot,
                   material=backfill, name="trench_backfill")
        objs.append(G._obj("trench_backfill", backfill,
                           box={"x_min": pit_cx - htop, "x_max": pit_cx + htop, "depth_top": 0.0, "depth_bottom": pit_bot}))
    else:                                                          # trapezoid / v: sloped walls (TU1208-like)
        hbot = htop * float(rng.uniform(0.55, 0.8))
        corners = [(pit_cx - htop, 0.0), (pit_cx + htop, 0.0), (pit_cx + hbot, pit_bot), (pit_cx - hbot, pit_bot)]
        sc.add_polygon(corners_xz=corners, material=backfill, name="trench_backfill")
        objs.append(G._obj("trench_backfill", backfill, polygon=corners,
                           box={"x_min": pit_cx - htop, "x_max": pit_cx + htop, "depth_top": 0.0, "depth_bottom": pit_bot}))
    notes.append(f"{shape} pit {pit_w:.0f}m ({backfill})")

    def _halfwidth_at(z):                                          # pit half-width at depth z (wall taper)
        return htop + (hbot - htop) * (z / pit_bot)

    # --- utilities in LAYERS inside the pit; materials vary (tu1208 = steel / water-PVC / empty-PVC) ---
    comps = sample_companions("utility_trench", rng, realistic=realistic_relation)   # provenance + extras
    rel0 = next((r for c, r, *_ in comps if c == "pipe"), ScenarioRelationType.SPATIAL_CONTAINED_IN)
    n_layers = 3 if is_tu else (int(rng.integers(2, 4)) if large else int(rng.integers(1, 3)))
    layer_depths = [0.3 + (pit_bot - 0.45) * (li + 0.5) / n_layers for li in range(n_layers)]
    tu_codes = ["steel", "water_pvc", "empty_pvc"]                 # TU1208 layer materials (Fig.9)
    duct_layer = int(rng.integers(0, n_layers)) if (not is_tu and rng.random() < 0.45) else -1
    for li, lz in enumerate(layer_depths):
        hw = _halfwidth_at(lz) * 0.82
        n_in = int(rng.integers(2, 5))                            # vary pipes-per-layer (locations + numbers)
        if li == duct_layer:                                      # a duct bank: concrete envelope + parallel conduits
            bw = float(min(2 * hw, 0.18 * n_in + 0.1)); dcx = pit_cx + float(rng.uniform(-0.3, 0.3)) * hw
            sc.add_box(x_min_m=dcx - bw / 2, x_max_m=dcx + bw / 2, depth_top_m=lz - 0.12, depth_bottom_m=lz + 0.12,
                       material="concrete", name="duct_bank_envelope")
            objs.append(G._obj("duct_bank_envelope", "concrete",
                               box={"x_min": dcx - bw / 2, "x_max": dcx + bw / 2, "depth_top": lz - 0.12, "depth_bottom": lz + 0.12}))
            _record("duct_bank_envelope", "conduit", Arrangement.PARALLEL_BAND, Arrangement.PARALLEL_BAND)
            cmat = str(rng.choice(["pvc", "air"]))
            for j in range(n_in):
                ox = (dcx - bw / 2 + 0.05 + (bw - 0.1) * (j + 0.5) / n_in) if realistic_relation else (dcx + float(rng.uniform(-hw, hw)))
                sc.add_void(center_x_m=float(ox), depth_m=lz, radius_m=0.035, material=cmat)
                objs.append(G._obj("conduit", cmat, x=float(ox), depth=lz, radius=0.035))
            notes.append(f"ductbank L{li}({n_in})")
            continue
        code = tu_codes[li % 3] if is_tu else str(rng.choice(["steel", "cast_iron", "pvc", "hdpe", "concrete"]))
        shell = "pvc" if code in ("water_pvc", "empty_pvc") else code
        for j in range(n_in):                                     # a parallel row of pipes (one layer)
            if realistic_relation:
                px = pit_cx - hw + (2 * hw) * (j + 0.5) / n_in; pz = lz + float(rng.uniform(-0.05, 0.05))
            else:                                                 # decorrelated -> scattered depth/x within the pit
                px = pit_cx + float(rng.uniform(-hw, hw)); pz = float(rng.uniform(0.5, pit_bot - 0.15))
            r = float(rng.uniform(0.04, 0.08))
            fill = ("water" if code == "water_pvc"                # water-filled / empty / random-empty plastic
                    else "air" if (code == "empty_pvc" or (shell in ("pvc", "hdpe", "concrete") and rng.random() < 0.5))
                    else None)
            ptype = ("metal" if shell in ("steel", "cast_iron") else "concrete" if shell == "concrete"
                     else "water_filled_plastic" if fill == "water" else "empty_plastic" if fill == "air"
                     else "solid_plastic")
            sc.add_pipe(center_x_m=float(px), depth_m=float(pz), radius_m=r, material=shell)
            po = G._obj("pipe", shell, x=float(px), depth=float(pz), radius=r)
            po["ptype"] = ptype; po["fill"] = fill or "none"      # explicit pipe TYPE + fill (separable)
            objs.append(po)
            if fill:
                sc.add_void(center_x_m=float(px), depth_m=float(pz), radius_m=r * 0.65, material=fill)
                objs.append(G._obj("pipe_fill_water" if fill == "water" else "pipe_void", fill,
                                   x=float(px), depth=float(pz), radius=r * 0.65))
    _record("utility_trench", "pipe", rel0, Arrangement.TRENCH_FLOOR if realistic_relation else Arrangement.SCATTER)
    notes.append(f"{n_layers} pipe layers" + (" [steel/water-PVC/empty-PVC]" if is_tu else ""))

    # --- a polystyrene cavity (always for TU1208; occasional elsewhere) ---
    if is_tu or rng.random() < 0.35:
        cav_x = float(np.clip(pit_cx + float(rng.uniform(-0.5, 0.5)) * htop, 0.6, W - 0.6))
        cav_z = float(rng.uniform(0.6, max(0.8, pit_bot - 0.2))); cav_r = float(rng.uniform(0.12, 0.25))
        # TU1208 polystyrene is a DOME; elsewhere pick any cavity shape. air ~ polystyrene (eps~1.05).
        _add_cavity(sc, rng, objs, cav_x, cav_z, cav_r, "air",
                    "polystyrene_cavity" if is_tu else "generic_void", shape="dome" if is_tu else None)
        notes.append("cavity")

    # --- surrounding host clutter OUTSIDE the pit (TU1208 gneiss blocks ~ granite) ---
    n_clut = int(rng.integers(2, 5)) if is_tu else (int(rng.integers(1, 4)) if large else int(rng.integers(0, 2)))
    for _ in range(n_clut):
        spans = [s for s in ((0.6, pit_cx - htop - 0.6), (pit_cx + htop + 0.6, W - 0.6)) if s[1] - s[0] > 0.5]
        if not spans:
            break
        s = spans[int(rng.integers(0, len(spans)))]
        x = float(rng.uniform(*s)); z = float(rng.uniform(0.5, D - 0.5))
        if is_tu or rng.random() < 0.6:
            rock = "granite" if is_tu else str(rng.choice(["granite", "limestone"])); rr = float(rng.uniform(0.08, 0.18))
            sc.add_pipe(center_x_m=x, depth_m=z, radius_m=rr, material=rock)
            objs.append(G._obj("gneiss_block" if is_tu else "boulder", rock, x=x, depth=z, radius=rr, ambiguity=True))
        else:
            _add_cavity(sc, rng, objs, x, z, float(rng.uniform(0.1, 0.22)), "air", "generic_void")

    meta = G._meta(SCENE_TYPE, soil, D, fc=fc, n_traces=0, note=f"{family}: " + ", ".join(notes))
    v = 3e8 / math.sqrt(max(G._eps(soil), 1.0))
    scan_step = SCAN_STEP_L_M if large else 0.03; tw_cap = TW_CAP_L_S if large else 6.0e-8
    meta.update(tier="L" if large else "M", scale="excavation" if large else "utility", family=family,
                dx_m=dx, scan_step_m=scan_step, tw_cap_s=tw_cap,
                time_window_s=float(min(tw_cap, 2.2 * (D + 0.2) / v + 5e-9)),
                coupling=coupling, antenna="hertzian_dipole", tx_rx_offset_m=tx_rx_offset,
                correlation_sampled=sampled, realistic_relation=realistic_relation)
    return sc, objs, meta


def _render_model_png(out_dir, meta, host):
    """Render the VOXELIZED eps_r model (what gprMax sees) from geometry.h5 + materials.txt."""
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eps = np.array([float(l.split()[1]) for l in open(out_dir / "materials.txt") if l.startswith("#material")])
    with h5py.File(out_dir / "geometry.h5", "r") as h:
        g = np.asarray(h["data"])[:, :, 0]; dx = float(h.attrs["dx_dy_dz"][0])
    gd = g[:, ::-1]                                            # flip y so row 0 = top of the domain
    js = next((j for j in range(gd.shape[1]) if (gd == 0).mean(0)[j] < 0.5), 0)   # first sub-surface row
    sub = eps[gd][:, js:]; Wm = sub.shape[0] * dx; Zm = sub.shape[1] * dx
    fig, ax = plt.subplots(figsize=(12, 3.4), dpi=120)
    im = ax.imshow(sub.T, origin="upper", aspect="auto", extent=[0, Wm, Zm, 0], cmap="turbo",
                   vmin=float(np.percentile(sub, 1)), vmax=float(np.percentile(sub, 99)))
    plt.colorbar(im, ax=ax, label="eps_r", fraction=0.03, pad=0.02)
    ax.set_title(f"{meta['scale']} | {meta['coupling']} | fc={meta['fc_hz']/1e6:.0f}MHz | host {host} "
                 f"(eps_r {meta['host_eps_r']}) | rel={'R' if meta['realistic_relation'] else 'D'} | {meta['note']}",
                 fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("depth (m)")
    fig.tight_layout(); fig.savefig(out_dir / "model.png"); plt.close(fig)


def run_composite_one(idx, seed, log, realistic_host_prob, realistic_relation_prob, *, dry=False, model_only=False):
    rng = np.random.default_rng(seed)
    host_rng = np.random.default_rng([seed, 0x484F5354])
    host = str(host_rng.choice(G.SOILS))                       # composite is host-agnostic
    realistic_relation = bool(host_rng.random() < realistic_relation_prob)
    sc, objs, meta = build_composite_corridor(rng, host_soil=host, realistic_relation=realistic_relation)
    # Drop pipe-only scenes: every scene must have non-pipe STRUCTURE (a pit, layers, duct, cavity, or
    # block) -- a bare field of pipes is not a useful composite sample.
    _ctx = {"pipe_void", "pipe_fill_water", "trench_backfill"}
    targets = {o["kind"] for o in objs if o["kind"] not in _ctx and not o["kind"].endswith(("_layer", "_stratum"))}
    if targets <= {"pipe"}:
        log(f"  SKIP idx={idx}: pipe-only scene (no non-pipe structure)")
        return None
    sc.soil_material = host; meta["host_material"] = host; meta["host_eps_r"] = round(G._eps(host), 3)
    het = {"eps_spread_frac": 0.10, "correlation_length_m": float(rng.uniform(0.1, 0.25)), "n_levels": 9, "seed": seed}
    sc.soil_heterogeneity = het

    scan_step = meta["scan_step_m"]; span = max(sc.width_m - 2 * G.SCAN_MARGIN_M, scan_step)
    n_traces = max(48, int(round(span / scan_step)) + 1)
    scan_start, tw_s = G.SCAN_MARGIN_M, meta["time_window_s"]
    cov_targets = G._coverage_targets(objs)
    covered = None
    if cov_targets:
        pre = G._coverage(cov_targets, scan_x_min_m=scan_start, scan_x_max_m=scan_start + (n_traces - 1) * scan_step,
                          host_eps_r=meta["host_eps_r"], fc_hz=meta["fc_hz"],
                          footprint_fn=G.footprint_half_width_m, n_traces=n_traces, time_window_ns=tw_s * 1e9)
        covered = pre["all_covered"]
        if not covered:
            scan_start = round(max(0.05, pre["recommended_scan_x_min_m"]), 4)
            scan_end = round(min(sc.width_m - 0.05, pre["recommended_scan_x_max_m"]), 4)
            n_traces = max(n_traces, int(round((scan_end - scan_start) / scan_step)) + 1,
                           int(pre.get("recommended_n_traces", n_traces)))
            scan_step = (scan_end - scan_start) / max(n_traces - 1, 1)
            tw_s = float(min(meta["tw_cap_s"], max(tw_s, pre["recommended_time_window_ns"] * 1e-9 * 1.1)))
            meta["time_window_s"] = tw_s
    meta["n_traces"] = n_traces

    card_comp = card_rels = None
    try:
        card, _ = CD.build_card_for_scene(card_id=f"comp{idx}", scene_type=SCENE_TYPE, soil=host,
                                          objs=objs, coupling=None, note=meta["note"])
        card_comp = card.scene_composition.value; card_rels = len(card.scene_relations)
    except Exception as e:
        log(f"  WARN domain card: {type(e).__name__}: {str(e)[:120]}")

    if dry:
        log(f"DRY {idx:02d} {meta['scale']:10s} {meta['coupling']:14s} fc={meta['fc_hz']/1e6:.0f}MHz "
            f"host={host:14s} rel={'R' if realistic_relation else 'D'} W={sc.width_m:4.1f} D={sc.soil_depth_m:3.1f} "
            f"objs={len(objs):2d} traces={n_traces:3d} tw={tw_s*1e9:4.0f}ns covered={covered} "
            f"comp={card_comp} rels={card_rels} | {meta['note']}")
        return None

    if model_only:                                            # voxelize + render the model, NO gprMax solve
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_dir = G.OUT_ROOT / SCENE_TYPE / f"composite_{ts}_{idx:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sc.write_gprmax(out_dir)                               # build123d -> OCP voxelize -> geometry.h5 + materials.txt
        _render_model_png(out_dir, meta, host)
        try:
            (out_dir / "card.json").write_text(card.to_json(), encoding="utf-8")
        except Exception:
            pass
        (out_dir / "labels.json").write_text(json.dumps({
            "scene_type": SCENE_TYPE, "tier": meta["tier"], "scale": meta["scale"], "coupling": meta["coupling"],
            "antenna": meta["antenna"], "host_material": host, "host_eps_r": meta["host_eps_r"],
            "scene_composition": card_comp, "realistic_relation": realistic_relation,
            "correlation_sampled": meta["correlation_sampled"], "objects": objs, "note": meta["note"],
            "acquisition": {"fc_hz": meta["fc_hz"], "n_traces": n_traces, "scan_step_m": round(scan_step, 5),
                            "time_window_s": meta["time_window_s"], "scene_width_m": sc.width_m, "model_only": True},
        }, indent=2), encoding="utf-8")
        log(f"MODEL {idx:02d} {meta['scale']:10s} {meta['coupling']:14s} fc={meta['fc_hz']/1e6:.0f}MHz "
            f"host={host:14s} rel={'R' if realistic_relation else 'D'} W={sc.width_m:.1f} D={sc.soil_depth_m:.1f} "
            f"objs={len(objs)} -> {out_dir.name}")
        return out_dir

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = G.OUT_ROOT / SCENE_TYPE / f"composite_{ts}_{idx:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = G.S.run_bscan(sc, out_dir=out_dir, fc_hz=meta["fc_hz"], n_traces=n_traces,
                      scan_start_m=scan_start, scan_step_m=scan_step, tx_rx_offset_m=meta["tx_rx_offset_m"],
                      time_window_s=tw_s, gpu=True)
    dt_ns, dx_m, b = r["dt_ns"], r["dx_m"], r["bscan"]
    G._save_preview(out_dir / "bscan.png", b)
    try:
        (out_dir / "card.json").write_text(card.to_json(), encoding="utf-8")
    except Exception:
        pass
    labels = {
        "scene_type": SCENE_TYPE, "tier": meta["tier"], "scale": meta["scale"], "coupling": meta["coupling"],
        "antenna": meta["antenna"], "tx_rx_offset_m": meta["tx_rx_offset_m"],
        "ambiguity": meta["ambiguity"], "note": meta["note"],
        "scene_composition": card_comp, "host_material": host, "host_eps_r": meta["host_eps_r"],
        "realistic_host_prob": realistic_host_prob, "realistic_relation": realistic_relation,
        "realistic_relation_prob": realistic_relation_prob,
        "correlation_sampled": meta["correlation_sampled"], "soil_heterogeneity": het, "objects": objs,
        "acquisition": {"fc_hz": meta["fc_hz"], "n_traces": n_traces, "dx_m": dx_m, "dt_ns": dt_ns,
                        "time_window_s": meta["time_window_s"], "scan_start_m": round(scan_start, 4),
                        "scan_step_m": round(scan_step, 5),
                        "scan_extent_m": [round(scan_start, 4), round(scan_start + (b.shape[1] - 1) * dx_m, 4)],
                        "scene_width_m": sc.width_m, "scan_width_m": round(dx_m * b.shape[1], 4)},
        "bscan_shape": list(b.shape), "material_order": r["material_order"],
    }
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    prov = {"engine": "gpr_agent.sim_scenes (build123d -> voxelize -> gprMax GPU)", "tier": "L",
            "seed": seed, "index": idx, "created_at": datetime.now(timezone.utc).isoformat(),
            "wall_s": round(time.time() - t0, 1), "meta": meta}
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    G._clean_caches(out_dir)
    with (G.OUT_ROOT / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"dir": str(out_dir.relative_to(G.OUT_ROOT)), "scene_type": SCENE_TYPE,
                            "tier": meta["tier"], "scale": meta["scale"], "coupling": meta["coupling"],
                            "antenna": meta["antenna"], "ambiguity": meta["ambiguity"], "n_objects": len(objs),
                            "n_relations": card_rels, "scene_composition": card_comp, "host": host,
                            "realistic_relation": realistic_relation, "realistic_relation_prob": realistic_relation_prob,
                            "bscan_shape": list(b.shape), "wall_s": prov["wall_s"], "created_at": prov["created_at"]}) + "\n")
    log(f"  OK composite {b.shape} host={host:14s}[{'R' if realistic_relation else 'D'}] "
        f"objs={len(objs)} rels={card_rels} {prov['wall_s']:.0f}s -> {out_dir.name}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--dry", action="store_true", help="build + validate scenes, no gprMax")
    ap.add_argument("--model-only", action="store_true", help="voxelize + render the eps model, no gprMax solve")
    ap.add_argument("--realistic-host-prob", type=float, default=1.0)
    ap.add_argument("--realistic-relation-prob", type=float, required=True,
                    help="prob in [0,1] that object<->object presence+arrangement is physical; else decorrelated")
    a = ap.parse_args()

    G.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logf = G.OUT_ROOT / "generation.log"

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"; print(line, flush=True)
        with logf.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    rng = np.random.default_rng(a.seed)
    deadline = time.time() + a.hours * 3600.0
    log(f"=== composite (Tier-L) corpus: n={a.n} dry={a.dry} relation_prob={a.realistic_relation_prob} ===")
    ok = fail = 0
    for idx in range(a.n):
        if time.time() >= deadline:
            break
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            run_composite_one(idx, seed, log, a.realistic_host_prob, a.realistic_relation_prob,
                               dry=a.dry, model_only=a.model_only)
            ok += 1
        except Exception as e:
            fail += 1
            log(f"  FAIL idx={idx}: {type(e).__name__}: {str(e)[:200]}")
            with (G.OUT_ROOT / "failures.log").open("a", encoding="utf-8") as f:
                f.write(f"\n=== composite idx={idx} seed={seed} ===\n" + traceback.format_exc())
    log(f"=== composite done: {ok} ok, {fail} failed ===")


if __name__ == "__main__":
    main()

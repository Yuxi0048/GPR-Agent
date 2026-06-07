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


def build_composite_corridor(rng, host_soil=None, realistic_relation=True):
    """A TU1208-class scene: a WIDE backfilled pit spanning most of the width, with utilities in
    2-3 LAYERS inside it, and blocks/cavities in the surrounding host. Object presence + layered
    arrangement come from object_correlation (strength-gated); the pit wall shape is host-correlated."""
    soil = host_soil or str(rng.choice(G.SOILS))
    # SCALE: size the canvas to the content -- a large excavation gets a large canvas; small utilities
    # get a small one (no point imaging a 0.6 m trench across 18 m).
    large = bool(rng.random() < 0.4)
    if large:
        W = float(rng.uniform(10.0, 18.0)); D = float(rng.uniform(3.0, 4.0)); dx = DX_L_M
        pit_frac = float(rng.uniform(0.55, 0.80))
    else:
        W = float(rng.uniform(3.0, 6.0)); D = float(rng.uniform(1.6, 2.6)); dx = 0.006
        pit_frac = float(rng.uniform(0.30, 0.60))
    sc = G.S.Scene(width_m=W, soil_depth_m=D, dx_m=dx, soil_material=soil)
    # COUPLING: the antenna sits in the air gap -> air_gap_m sets ground-coupled vs air-launched.
    air_launched = bool(rng.random() < 0.3)
    sc.air_gap_m = float(rng.uniform(0.25, 0.5)) if air_launched else float(rng.uniform(0.02, 0.06))
    coupling = "air_launched" if air_launched else "ground_coupled"
    # ANTENNA: the 2-D corpus uses a hertzian dipole; vary fc by scale + the Tx-Rx offset. (The 3-D
    # MALA antenna is a separate ~50-100x slower path -- temp_src/run_mala_3d.py -- not generated here.)
    fc = 2.5e8 if large else float(rng.choice([5.0e8, 8.0e8])); tx_rx_offset = round(float(rng.uniform(0.04, 0.18)), 3)
    objs: list[dict] = []
    notes: list[str] = []
    sampled: list[dict] = []

    def _record(anchor, companion, relation, arr):
        sampled.append({"anchor": anchor, "companion": companion,
                        "relation": relation.value, "arrangement": arr.value})

    # --- wide backfilled excavation spanning most of the width; wall shape host-correlated ---
    backfill = str(rng.choice(["dry_sand", "gravel", "silt", "moist_limestone"]))
    pit_w = pit_frac * W
    pit_cx = float(np.clip(W / 2 + rng.uniform(-0.08, 0.08) * W, pit_w / 2 + 0.3, W - pit_w / 2 - 0.3))
    pit_bot = float(rng.uniform(min(1.2, D - 0.6), D - 0.3)); htop = pit_w / 2
    shape = correlated_shape_for("utility_trench", soil, rng, realistic=realistic_relation)   # OSHA/EC7
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

    # --- utilities in 2-3 LAYERS inside the pit (trench->pipe co-occurrence; layered arrangement) ---
    comps = sample_companions("utility_trench", rng, realistic=realistic_relation)   # provenance + extras
    if True:                                                  # the excavation is dug FOR utilities -> always present
        rel0 = next((r for c, r, *_ in comps if c == "pipe"), ScenarioRelationType.SPATIAL_CONTAINED_IN)
        n_layers = int(rng.integers(2, 4)) if large else int(rng.integers(1, 3))
        layer_depths = [0.3 + (pit_bot - 0.45) * (li + 0.5) / n_layers for li in range(n_layers)]
        duct_layer = int(rng.integers(0, n_layers)) if rng.random() < 0.5 else -1   # one layer may be a duct bank
        for li, lz in enumerate(layer_depths):
            hw = _halfwidth_at(lz) * 0.82
            n_in = int(rng.integers(2, 5))
            if li == duct_layer:                                  # a duct bank: concrete envelope + parallel conduits
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
            mat = str(rng.choice(["steel", "cast_iron", "pvc", "hdpe", "concrete"]))
            for j in range(n_in):                                 # a parallel row of pipes (one layer)
                if realistic_relation:
                    px = pit_cx - hw + (2 * hw) * (j + 0.5) / n_in; pz = lz + float(rng.uniform(-0.05, 0.05))
                else:                                             # decorrelated -> scattered depth/x within the pit
                    px = pit_cx + float(rng.uniform(-hw, hw)); pz = float(rng.uniform(0.5, pit_bot - 0.15))
                r = float(rng.uniform(0.04, 0.08))
                sc.add_pipe(center_x_m=float(px), depth_m=float(pz), radius_m=r, material=mat)
                objs.append(G._obj("pipe", mat, x=float(px), depth=float(pz), radius=r))
                if mat in ("pvc", "hdpe", "concrete") and rng.random() < 0.5:    # empty pipe -> inner air
                    sc.add_void(center_x_m=float(px), depth_m=float(pz), radius_m=r * 0.65, material="air")
                    objs.append(G._obj("pipe_void", "air", x=float(px), depth=float(pz), radius=r * 0.65))
        _record("utility_trench", "pipe", rel0, Arrangement.TRENCH_FLOOR if realistic_relation else Arrangement.SCATTER)
        notes.append(f"{n_layers} pipe layers")

    # --- surrounding host clutter OUTSIDE the pit (gneiss blocks / cavities, TU1208-like) ---
    for _ in range(int(rng.integers(1, 4)) if large else int(rng.integers(0, 2))):
        spans = [s for s in ((0.6, pit_cx - htop - 0.6), (pit_cx + htop + 0.6, W - 0.6)) if s[1] - s[0] > 0.5]
        if not spans:
            break
        s = spans[int(rng.integers(0, len(spans)))]
        x = float(rng.uniform(*s)); z = float(rng.uniform(0.5, D - 0.5))
        if rng.random() < 0.6:
            rock = str(rng.choice(["granite", "limestone"])); rr = float(rng.uniform(0.08, 0.18))
            sc.add_pipe(center_x_m=x, depth_m=z, radius_m=rr, material=rock)
            objs.append(G._obj("boulder", rock, x=x, depth=z, radius=rr, ambiguity=True))
        else:
            rr = float(rng.uniform(0.1, 0.22))
            sc.add_void(center_x_m=x, depth_m=z, radius_m=rr, material="air")
            objs.append(G._obj("generic_void", "air", x=x, depth=z, radius=rr, ambiguity=True))

    meta = G._meta(SCENE_TYPE, soil, D, fc=fc, n_traces=0, note="corridor: " + ", ".join(notes))
    v = 3e8 / math.sqrt(max(G._eps(soil), 1.0))
    scan_step = SCAN_STEP_L_M if large else 0.03; tw_cap = TW_CAP_L_S if large else 6.0e-8
    meta.update(tier="L" if large else "M", scale="excavation" if large else "utility",
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

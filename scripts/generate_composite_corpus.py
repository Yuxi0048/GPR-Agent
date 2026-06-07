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
from subsurface_platform.domain.object_correlation import (
    CLUTTER_COMPANIONS, Arrangement, sample_companions)

SCENE_TYPE = "composite_corridor"
SCAN_STEP_L_M = 0.05            # coarser line spacing for 250 MHz (anti-alias-checked below)
DX_L_M = 0.01                  # 10 mm voxel (TU1208 scale)
TW_CAP_L_S = 1.5e-7            # up to 150 ns (deep scene)


def build_composite_corridor(rng, host_soil=None, realistic_relation=True):
    """A large multi-object utility corridor; presence + arrangement from object_correlation."""
    soil = host_soil or str(rng.choice(G.SOILS))
    W = float(rng.uniform(12.0, 20.0)); D = float(rng.uniform(3.0, 4.5))
    sc = G.S.Scene(width_m=W, soil_depth_m=D, dx_m=DX_L_M, soil_material=soil)
    objs: list[dict] = []
    notes: list[str] = []
    sampled: list[dict] = []                                   # correlation provenance

    def _record(anchor, companion, relation, arr):
        sampled.append({"anchor": anchor, "companion": companion,
                        "relation": relation.value, "arrangement": arr.value})

    # --- primary: a utility trench with bedded pipe(s) (TRENCH_FLOOR arrangement) ---
    backfill = str(rng.choice(["dry_sand", "gravel", "silt"]))
    tw_ = float(rng.uniform(0.8, 1.4)); tcx = float(rng.uniform(2.0, W * 0.45)); tbot = float(rng.uniform(1.2, 2.2))
    sc.add_box(x_min_m=tcx - tw_ / 2, x_max_m=tcx + tw_ / 2, depth_top_m=0.0, depth_bottom_m=tbot,
               material=backfill, name="trench_backfill")
    objs.append(G._obj("trench_backfill", backfill,
                       box={"x_min": tcx - tw_ / 2, "x_max": tcx + tw_ / 2, "depth_top": 0.0, "depth_bottom": tbot}))
    for companion, relation, n, arr in sample_companions("utility_trench", rng, realistic=realistic_relation):
        if companion != "pipe":
            continue
        _record("utility_trench", companion, relation, arr)
        for _ in range(n):
            mat = str(rng.choice(["pvc", "steel", "cast_iron", "concrete"])); r = float(rng.uniform(0.04, 0.08))
            if arr is Arrangement.TRENCH_FLOOR:
                px = float(rng.uniform(tcx - tw_ / 2 + 0.15, tcx + tw_ / 2 - 0.15)); pz = tbot - float(rng.uniform(0.1, 0.3))
            else:                                              # decorrelated -> anywhere
                px = float(rng.uniform(1.0, W - 1.0)); pz = float(rng.uniform(0.5, D - 0.5))
            sc.add_pipe(center_x_m=px, depth_m=pz, radius_m=r, material=mat)
            objs.append(G._obj("pipe", mat, x=px, depth=pz, radius=r))
    notes.append(f"trench({backfill})")

    # --- secondary: a parallel-conduit duct bank in another lateral location / depth band ---
    if rng.random() < 0.85:
        cols = int(rng.integers(2, 6)); cw = 0.12; bw = cols * cw + 0.08; bh = 0.25
        dcx = float(rng.uniform(W * 0.55, W - bw / 2 - 1.0)); dtop = float(rng.uniform(0.6, 1.6))
        sc.add_box(x_min_m=dcx - bw / 2, x_max_m=dcx + bw / 2, depth_top_m=dtop, depth_bottom_m=dtop + bh,
                   material="concrete", name="duct_bank_envelope")
        objs.append(G._obj("duct_bank_envelope", "concrete",
                           box={"x_min": dcx - bw / 2, "x_max": dcx + bw / 2, "depth_top": dtop, "depth_bottom": dtop + bh}))
        for companion, relation, n, arr in sample_companions("duct_bank_envelope", rng, realistic=realistic_relation):
            if companion != "conduit":
                continue
            _record("duct_bank_envelope", companion, relation, arr)
            cmat = str(rng.choice(["pvc", "air"]))
            for i in range(n):
                if arr is Arrangement.PARALLEL_BAND:
                    ox = dcx - bw / 2 + 0.06 + i * (bw - 0.12) / max(n - 1, 1); oz = dtop + bh / 2
                else:
                    ox = float(rng.uniform(1.0, W - 1.0)); oz = float(rng.uniform(0.5, D - 0.5))
                sc.add_void(center_x_m=ox, depth_m=oz, radius_m=0.04, material=cmat)
                objs.append(G._obj("conduit", cmat, x=ox, depth=oz, radius=0.04))
        notes.append(f"ductbank({cols})")

    # --- deep band: a crossing / deep service ---
    if rng.random() < 0.6:
        mat = str(rng.choice(["steel", "cast_iron", "pvc"])); r = float(rng.uniform(0.05, 0.09))
        px = float(rng.uniform(2.0, W - 2.0)); pz = float(rng.uniform(2.4, D - 0.4))
        sc.add_pipe(center_x_m=px, depth_m=pz, radius_m=r, material=mat)
        objs.append(G._obj("pipe", mat, x=px, depth=pz, radius=r))
        notes.append("deep_service")

    # --- host clutter (a couple of boulders / voids) ---
    for _ in range(int(rng.integers(0, 3))):
        ck = str(rng.choice(CLUTTER_COMPANIONS))
        x = float(rng.uniform(1.0, W - 1.0)); z = float(rng.uniform(0.5, D - 0.5))
        if ck == "boulder":
            rock = str(rng.choice(["granite", "limestone"])); rr = float(rng.uniform(0.06, 0.15))
            sc.add_pipe(center_x_m=x, depth_m=z, radius_m=rr, material=rock)
            objs.append(G._obj("boulder", rock, x=x, depth=z, radius=rr, ambiguity=True))
        elif ck == "generic_void":
            rr = float(rng.uniform(0.08, 0.2))
            sc.add_void(center_x_m=x, depth_m=z, radius_m=rr, material="air")
            objs.append(G._obj("generic_void", "air", x=x, depth=z, radius=rr, ambiguity=True))

    meta = G._meta(SCENE_TYPE, soil, D, fc=2.5e8, n_traces=0, note="corridor: " + ", ".join(notes))
    v = 3e8 / math.sqrt(max(G._eps(soil), 1.0))
    meta.update(tier="L", dx_m=DX_L_M, scan_step_m=SCAN_STEP_L_M, tw_cap_s=TW_CAP_L_S,
                time_window_s=float(min(TW_CAP_L_S, 2.2 * (D + 0.2) / v + 5e-9)),
                correlation_sampled=sampled, realistic_relation=realistic_relation)
    return sc, objs, meta


def run_composite_one(idx, seed, log, realistic_host_prob, realistic_relation_prob, *, dry=False):
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
        log(f"DRY {idx:02d} host={host:14s} rel={'R' if realistic_relation else 'D'} "
            f"W={sc.width_m:4.1f} D={sc.soil_depth_m:3.1f} objs={len(objs):2d} traces={n_traces:3d} "
            f"tw={tw_s*1e9:4.0f}ns covered={covered} comp={card_comp} rels={card_rels} | {meta['note']}")
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = G.OUT_ROOT / SCENE_TYPE / f"composite_{ts}_{idx:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = G.S.run_bscan(sc, out_dir=out_dir, fc_hz=meta["fc_hz"], n_traces=n_traces,
                      scan_start_m=scan_start, scan_step_m=scan_step, time_window_s=tw_s, gpu=True)
    dt_ns, dx_m, b = r["dt_ns"], r["dx_m"], r["bscan"]
    G._save_preview(out_dir / "bscan.png", b)
    try:
        (out_dir / "card.json").write_text(card.to_json(), encoding="utf-8")
    except Exception:
        pass
    labels = {
        "scene_type": SCENE_TYPE, "tier": "L", "ambiguity": meta["ambiguity"], "note": meta["note"],
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
                            "tier": "L", "ambiguity": meta["ambiguity"], "n_objects": len(objs),
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
            run_composite_one(idx, seed, log, a.realistic_host_prob, a.realistic_relation_prob, dry=a.dry)
            ok += 1
        except Exception as e:
            fail += 1
            log(f"  FAIL idx={idx}: {type(e).__name__}: {str(e)[:200]}")
            with (G.OUT_ROOT / "failures.log").open("a", encoding="utf-8") as f:
                f.write(f"\n=== composite idx={idx} seed={seed} ===\n" + traceback.format_exc())
    log(f"=== composite done: {ok} ok, {fail} failed ===")


if __name__ == "__main__":
    main()

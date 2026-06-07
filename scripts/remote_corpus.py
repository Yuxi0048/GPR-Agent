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
    return scan_start, scan_step, n_traces, tw_s


def prepare(n, seed, rhp, rrp, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); made = 0
    for idx in range(n):
        s = int(rng.integers(0, 2**31 - 1))
        srng = np.random.default_rng(s); host_rng = np.random.default_rng([s, 0x484F5354])
        host = str(host_rng.choice(G.SOILS)); rr = bool(host_rng.random() < rrp)
        sc, objs, meta = GC.build_composite_corridor(srng, host_soil=host, realistic_relation=rr)
        ctx = {"pipe_void", "pipe_fill_water", "trench_backfill"}
        tk = {o["kind"] for o in objs if o["kind"] not in ctx and not o["kind"].endswith(("_layer", "_stratum"))}
        if tk <= {"pipe"}:
            print(f"  skip {idx}: pipe-only"); continue
        sc.soil_material = host; meta["host_material"] = host; meta["host_eps_r"] = round(G._eps(host), 3)
        sc.soil_heterogeneity = {"eps_spread_frac": 0.10, "correlation_length_m": float(srng.uniform(0.1, 0.25)),
                                 "n_levels": 9, "seed": s}
        scan_start, scan_step, n_traces, tw_s = _acquisition(sc, objs, meta)
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
             "realistic_relation": rr}, default=str, indent=2), encoding="utf-8")
        made += 1
        print(f"PREP {idx:02d} {meta.get('family','?'):10s} {meta['scale']:10s} host={host:14s} "
              f"traces={n_traces:3d} -> {d.name}")
    print(f"PREPARE_DONE made={made} out={out}")


def assemble(out):
    import h5py
    out = Path(out); OUT = G.OUT_ROOT / GC.SCENE_TYPE; OUT.mkdir(parents=True, exist_ok=True); done = 0
    for d in sorted(out.glob("scene_*")):
        pj = json.loads((d / "prep.json").read_text())
        nt = pj["n_traces"]; sol = {}; dt = None
        for k in range(nt):
            o = d / f"t{k:03d}.out"
            if not o.exists():
                continue
            with h5py.File(o, "r") as f:
                dt = float(f.attrs["dt"]); sol[k] = np.asarray(f["rxs"]["rx1"]["Ez"], float)
        if len(sol) < 0.9 * nt:                                   # need most traces; else not worth assembling
            print(f"  skip {d.name}: only {len(sol)}/{nt} traces solved"); continue
        ns = len(next(iter(sol.values()))); b = np.zeros((ns, nt))
        for k, tr in sol.items():
            b[:, k] = tr
        if len(sol) < nt:                                         # pad the few missing with a neighbour
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
                  "host_eps_r": meta["host_eps_r"], "scene_composition": pj["scene_composition"],
                  "realistic_relation": pj["realistic_relation"], "correlation_sampled": meta["correlation_sampled"],
                  "objects": pj["objects"], "note": meta["note"],
                  "acquisition": {"fc_hz": meta["fc_hz"], "n_traces": nt, "dx_m": dx, "dt_ns": dt * 1e9,
                                  "time_window_s": pj["tw_s"], "scan_start_m": pj["scan_start"],
                                  "scan_step_m": pj["scan_step"], "scene_width_m": pj["width"]},
                  "bscan_shape": list(b.shape), "material_order": pj["material_order"], "solve": "remote_gpu_a100"}
        (sd / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
        with (G.OUT_ROOT / "manifest.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"dir": str(sd.relative_to(G.OUT_ROOT)), "scene_type": GC.SCENE_TYPE, "tier": meta["tier"],
                                "scale": meta["scale"], "family": meta.get("family"), "coupling": meta["coupling"],
                                "n_objects": len(pj["objects"]), "scene_composition": pj["scene_composition"],
                                "host": pj["host"], "realistic_relation": pj["realistic_relation"],
                                "bscan_shape": list(b.shape), "solve": "remote_gpu_a100", "created_at": ts}) + "\n")
        done += 1
        print(f"ASSEMBLE {d.name} -> {sd.name}  bscan {b.shape}")
    print(f"ASSEMBLE_DONE n={done}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "assemble"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realistic-host-prob", type=float, default=1.0)
    ap.add_argument("--realistic-relation-prob", type=float, default=0.7)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare(a.n, a.seed, a.realistic_host_prob, a.realistic_relation_prob, a.out)
    else:
        assemble(a.out)


if __name__ == "__main__":
    main()

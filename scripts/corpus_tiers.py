"""Two-tier subsurface scene generation — see docs/corpus-two-tier-design.md.

Tier T (Template): the trusted parametric BUILDERS in generate_subsurface_corpus.py — run
unattended -> generated_corpus/. Reliable by contract (seeded, labels-by-construction, Gate A/B).

Tier F (Freeform): a natural-language description -> candidate Scene -> Gate A -> a review
bundle in proposals/<id>/ (status=pending). It NEVER reaches the corpus without a recorded human
decision. Promotion turns an approved candidate into a Tier-T template.

The freeform NL->scene here is a self-contained lightweight parser; the production upgrade is
subsurface_platform.extraction.CompositeExtractor (noted at the seam).

CLI:
    python corpus_tiers.py registry                      # (re)write the Tier-T registry.json
    python corpus_tiers.py propose "<description>" [--seed N]
    python corpus_tiers.py list
    python corpus_tiers.py review <id> {reject|accept_sample|promote} [--notes "..."] [--by NAME]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), r"e:/github/GPR-Agent/src", r"e:/github/GPR-Sim/src",
                r"e:/github/GPR-KnowledgeBase", r"e:/github/GPR-Tools/src"]

import gpr_agent.sim_scenes as S
from generate_subsurface_corpus import BUILDERS, _eps, _is_conductor, _obj  # Tier-T templates

DATA = Path("e:/github/GPR-Sim/data")
CORPUS = DATA / "generated_corpus"
PROPOSALS = DATA / "proposals"
REGISTRY = DATA / "templates" / "registry.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Tier T — the trusted template registry
# --------------------------------------------------------------------------- #
def write_registry() -> Path:
    """Snapshot the trusted BUILDERS into registry.json with provenance."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if REGISTRY.exists():
        existing = {t["name"]: t for t in json.loads(REGISTRY.read_text()).get("templates", [])}
    templates = []
    for name in BUILDERS:
        prev = existing.get(name, {})
        templates.append({
            "name": name,
            "origin": prev.get("origin", "native"),         # native | promoted_from_freeform:<id>
            "version": prev.get("version", 1),
            "reviewed_by": prev.get("reviewed_by", "design"),
            "reviewed_at": prev.get("reviewed_at", _now()),
        })
    REGISTRY.write_text(json.dumps({"updated_at": _now(), "templates": templates}, indent=2),
                        encoding="utf-8")
    return REGISTRY


def register_template(name: str, *, origin: str, reviewed_by: str) -> None:
    """Add/replace a template entry (used by promotion)."""
    write_registry()
    reg = json.loads(REGISTRY.read_text())
    reg["templates"] = [t for t in reg["templates"] if t["name"] != name]
    reg["templates"].append({"name": name, "origin": origin, "version": 1,
                             "reviewed_by": reviewed_by, "reviewed_at": _now()})
    reg["updated_at"] = _now()
    REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Tier F — freeform: natural language -> candidate Scene
# --------------------------------------------------------------------------- #
_HOSTS = ["dry_sand", "saturated_sand", "wet_clay", "dry_clay", "moist_limestone",
          "silt", "loam", "topsoil_moist"]
_KIND_MAT = [                                   # (keywords, kind, material)
    (("steel", "metal", "iron", "pec", "cast iron"), "pipe", "pec"),
    (("hdpe",), "pipe", "hdpe"),
    (("pvc", "plastic"), "pipe", "pvc"),
    (("concrete pipe",), "pipe", "concrete"),
    (("void", "cavity", "air pocket", "empty"), "void", "air"),
    (("boulder", "cobble", "rock", "granite"), "boulder", "granite"),
    (("rebar", "reinforc", "mesh"), "rebar", "pec"),
    (("root",), "tree_root", "root"),
]


def _num(pat: str, text: str, default=None):
    m = re.search(pat, text)
    return float(m.group(1)) if m else default


def freeform_to_scene(description: str, rng):
    """Lightweight NL -> Scene. Returns (scene, label_objects, parse_report).

    Production seam: replace this with
    ``subsurface_platform.extraction.CompositeExtractor`` for richer extraction.
    """
    text = " " + description.lower().strip() + " "
    host = next((h for h in _HOSTS if h.replace("_", " ") in text or h in text), "dry_sand")
    W = float(_num(r"width\s*=?\s*([0-9.]+)", text, 1.3))
    D = float(_num(r"depth\s*(?:of\s*)?(?:domain|soil)?\s*=?\s*([0-9.]+)", text, 0.9))
    sc = S.Scene(width_m=W, soil_depth_m=D, dx_m=0.005, soil_material=host)
    sc.soil_heterogeneity = {"eps_spread_frac": 0.10, "correlation_length_m": 0.10,
                             "n_levels": 9, "seed": int(rng.integers(0, 2**31 - 1))}
    objs, report = [], {"host": host, "width_m": W, "depth_m": D, "parsed": [], "unparsed": []}

    # trench (box | trapezoid | v_shape) + optional topsoil cap
    if "trench" in text:
        shape = ("trapezoid" if "trapezoid" in text else "v_shape" if (" v " in text or "v-shape" in text or "v shape" in text)
                 else "box")
        backfill = next((b for b in ("gravel", "dry_sand", "silt") if b.replace("_", " ") in text), "gravel")
        cx = W / 2; htop = rng.uniform(0.22, 0.30); bottom = rng.uniform(0.5, min(0.72, D - 0.1))
        if shape == "box":
            sc.add_box(x_min_m=cx - htop, x_max_m=cx + htop, depth_top_m=0.0, depth_bottom_m=bottom,
                       material=backfill, name="trench_backfill")
            objs.append(_obj("trench_backfill", backfill, box={"x_min": cx - htop, "x_max": cx + htop,
                                                               "depth_top": 0.0, "depth_bottom": bottom}))
        else:
            hbot = htop * (0.6 if shape == "trapezoid" else 0.0)
            corners = ([(cx - htop, 0.0), (cx + htop, 0.0), (cx + hbot, bottom), (cx - hbot, bottom)]
                       if shape == "trapezoid" else [(cx - htop, 0.0), (cx + htop, 0.0), (cx, bottom)])
            sc.add_polygon(corners_xz=corners, material=backfill, name="trench_backfill")
            objs.append(_obj("trench_backfill", backfill, polygon=corners))
        report["parsed"].append(f"trench:{shape}:{backfill}")
        if "topsoil" in text or "resurfac" in text or "capped" in text:
            sc.add_layer(depth_top_m=0.0, thickness_m=0.12, material="topsoil_moist")
            objs.append(_obj("topsoil_cap", "topsoil_moist",
                             box={"x_min": 0.0, "x_max": round(W, 4), "depth_top": 0.0, "depth_bottom": 0.12}))
            report["parsed"].append("topsoil_cap")

    # discrete targets, one per clause
    clauses = re.split(r"[;,]|\band\b|(?<!\d)\.(?!\d)", text)   # split on , ; "and" + sentence-dots, NOT decimals
    i = 0
    for cl in clauses:
        hit = next((km for km in _KIND_MAT if any(w in cl for w in km[0])), None)
        if hit is None:
            if cl.strip() and not any(t in cl for t in ("trench", "topsoil", host, "width", "depth")):
                report["unparsed"].append(cl.strip())
            continue
        _, kind, mat = hit
        x = _num(r"x\s*=?\s*([0-9.]+)", cl) or _num(r"at\s+([0-9.]+)\s*m", cl) or (0.3 + i * 0.4)
        z = _num(r"([0-9.]+)\s*m?\s*deep", cl) or _num(r"depth\s*=?\s*([0-9.]+)", cl) or rng.uniform(0.2, 0.5)
        r = _num(r"radius\s*=?\s*([0-9.]+)", cl) or _num(r"([0-9.]+)\s*cm", cl) or (rng.uniform(0.03, 0.07))
        x = min(max(float(x), 0.15), W - 0.15); z = min(float(z), D - 0.1); r = min(float(r), 0.12)
        sc.add_pipe(center_x_m=x, depth_m=z, radius_m=r, material=mat)
        objs.append(_obj(kind, mat, x=x, depth=z, radius=r,
                         ambiguity=kind in ("boulder", "tree_root", "rebar")))
        report["parsed"].append(f"{kind}:{mat}@x{x:.2f},z{z:.2f},r{r:.3f}")
        i += 1
    return sc, objs, report


# --------------------------------------------------------------------------- #
# Gate A (auto) + previews
# --------------------------------------------------------------------------- #
def gate_a(scene) -> dict:
    """CAD validity: each solid OCP-valid, non-degenerate, inside the domain."""
    per, issues = [], []
    for ob in scene.objects:
        prob = []
        try:
            iv = ob.solid.is_valid
            iv = iv() if callable(iv) else iv
            if not bool(iv):
                prob.append("OCP reports invalid solid")
        except Exception as e:                                          # pragma: no cover
            prob.append(f"validity check raised: {e}")
        try:
            bb = ob.solid.bounding_box()
            x0, y0, _ = bb.min.to_tuple(); x1, y1, _ = bb.max.to_tuple()
            if (x1 - x0) <= 0 or (y1 - y0) <= 0:
                prob.append("degenerate bbox")
            if x0 < -1e-6 or x1 > scene.width_m + 1e-6:
                prob.append("outside domain in x")
        except Exception as e:                                         # pragma: no cover
            prob.append(f"bbox raised: {e}")
        per.append({"name": ob.name, "material": ob.material, "ok": not prob, "issues": prob})
        issues += prob
    return {"ok": not issues, "n_issues": len(issues), "objects": per}


def _eps_map_png(scene, path: Path) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import h5py, tempfile
    with tempfile.TemporaryDirectory() as td:
        scene.write_gprmax(Path(td))                                  # OCP voxelize (no gprMax solve)
        eps = np.array([float(l.split()[1]) for l in open(Path(td) / "materials.txt") if l.startswith("#material")])
        with h5py.File(Path(td) / "geometry.h5", "r") as h:
            g = np.asarray(h["data"])[:, :, 0]; dx = float(h.attrs["dx_dy_dz"][0])
    gd = g[:, ::-1]; js = next((j for j in range(gd.shape[1]) if (gd == 0).mean(0)[j] < 0.5), 0)
    epsg = eps[gd][:, js:]
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110)
    im = ax.imshow(epsg.T, origin="upper", aspect="auto", extent=[0, gd.shape[0] * dx, epsg.shape[1] * dx, 0],
                   cmap="turbo", vmin=float(epsg.min()), vmax=float(np.percentile(epsg, 99)))
    plt.colorbar(im, ax=ax, label="εr", fraction=0.045)
    ax.set_title("freeform proposal — permittivity model"); ax.set_xlabel("x (m)"); ax.set_ylabel("depth (m)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _radargram_png(scene, path: Path) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from subsurface_platform.simulation import kirchhoff_forward, KirchhoffAcquisition
    host_eps = _eps(scene.soil_material)
    pts = []
    for ob in scene.objects:
        bb = ob.solid.bounding_box(); x0, y0, _ = bb.min.to_tuple(); x1, y1, _ = bb.max.to_tuple()
        cx = 0.5 * (x0 + x1); depth = scene.soil_depth_m - 0.5 * (y0 + y1)
        pts.append({"type": "pipe", "center_x": cx, "center_z": depth,
                    "eps_r": _eps(ob.material), "conductor": _is_conductor(ob.material)})
    ntr = max(40, int(scene.width_m / 0.02))
    acq = KirchhoffAcquisition(n_traces=ntr, trace_step_m=scene.width_m / ntr, start_x_m=0.0,
                               n_samples=420, dt_ns=0.08, host_eps_r=host_eps, center_freq_ghz=0.8)
    b = kirchhoff_forward(pts, acq, specular_power=2.0)
    d = np.clip(b / (np.percentile(np.abs(b), 99) + 1e-12), -1, 1)
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110)
    ax.imshow(d, cmap="gray", aspect="auto", vmin=-1, vmax=1,
              extent=[0, scene.width_m, acq.n_samples * acq.dt_ns, 0])
    ax.set_title("freeform proposal — fast Kirchhoff radargram (preview)")
    ax.set_xlabel("distance (m)"); ax.set_ylabel("two-way time (ns)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Propose / review / promote
# --------------------------------------------------------------------------- #
def propose_freeform(description: str, *, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    sc, objs, report = freeform_to_scene(description, rng)
    ga = gate_a(sc)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    pid = f"prop_{ts}_{seed:04d}"
    d = PROPOSALS / pid; d.mkdir(parents=True, exist_ok=True)
    (d / "scene.json").write_text(json.dumps(
        {"description": description, "seed": seed, "host": sc.soil_material,
         "width_m": sc.width_m, "soil_depth_m": sc.soil_depth_m, "parse_report": report}, indent=2), encoding="utf-8")
    (d / "labels.json").write_text(json.dumps({"objects": objs, "host_material": sc.soil_material}, indent=2), encoding="utf-8")
    (d / "gate_a.json").write_text(json.dumps(ga, indent=2), encoding="utf-8")
    try:
        _eps_map_png(sc, d / "model.png")
        _radargram_png(sc, d / "radargram.png")
    except Exception as e:                                             # preview is best-effort
        (d / "preview_error.txt").write_text(str(e), encoding="utf-8")
    review = {"proposal_id": pid, "status": "pending", "reviewer": None, "reviewed_at": None,
              "gate_a_ok": ga["ok"], "checks": {}, "notes": "", "promotion": None}
    (d / "review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    with (PROPOSALS / "proposals_manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"proposal_id": pid, "status": "pending", "gate_a_ok": ga["ok"],
                            "n_objects": len(objs), "host": sc.soil_material,
                            "description": description, "created_at": _now()}) + "\n")
    print(f"PROPOSED {pid}  gate_a_ok={ga['ok']}  objs={len(objs)}  parsed={report['parsed']}"
          + (f"  UNPARSED={report['unparsed']}" if report["unparsed"] else ""))
    print(f"  review bundle: {d}  (model.png, radargram.png, labels.json, gate_a.json)")
    return d


def _scaffold_template(pid: str, labels: dict, scene_meta: dict) -> Path:
    """Promotion: emit a parametric build_<name>(rng) stub from the reviewed candidate."""
    name = f"promoted_{pid.split('_')[1]}"
    out = _HERE / "promoted_templates"; out.mkdir(parents=True, exist_ok=True)
    lines = [f'"""Promoted from freeform proposal {pid} (human-reviewed). Finalize the parameter',
             '   ranges (rng.uniform/choice) then register in the Tier-T registry."""',
             "import gpr_agent.sim_scenes as S",
             "from generate_subsurface_corpus import _obj, _meta", "",
             f"def build_{name}(rng):",
             f"    sc = S.Scene(width_m={scene_meta['width_m']:.3f}, soil_depth_m={scene_meta['soil_depth_m']:.3f},"
             f" dx_m=0.005, soil_material={scene_meta['host']!r})",
             "    objs = []",
             "    # --- candidate objects (parameterize the literals with rng) ---"]
    for o in labels["objects"]:
        if "polygon_m" in o:
            lines.append(f"    sc.add_polygon(corners_xz={o['polygon_m']}, material={o['material']!r}, name={o['kind']!r})")
        elif "box_m" in o:
            b = o["box_m"]
            lines.append(f"    sc.add_box(x_min_m={b['x_min']}, x_max_m={b['x_max']}, depth_top_m={b['depth_top']},"
                         f" depth_bottom_m={b['depth_bottom']}, material={o['material']!r}, name={o['kind']!r})")
        elif "center_x_m" in o:
            lines.append(f"    sc.add_pipe(center_x_m={o['center_x_m']}, depth_m={o['depth_m']},"
                         f" radius_m={o['radius_m']}, material={o['material']!r})")
        lines.append(f"    objs.append(_obj({o['kind']!r}, {o['material']!r}))")
    lines += [f"    return sc, objs, _meta('{name}', {scene_meta['host']!r}, {scene_meta['soil_depth_m']:.3f},"
              f" note='promoted from {pid}')", ""]
    p = out / f"build_{name}.py"; p.write_text("\n".join(lines), encoding="utf-8")
    register_template(name, origin=f"promoted_from_freeform:{pid}", reviewed_by="review_cli")
    return p


def review(proposal_id: str, decision: str, *, reviewer: str = "human", notes: str = "") -> None:
    d = PROPOSALS / proposal_id
    rv = json.loads((d / "review.json").read_text())
    assert decision in ("reject", "accept_sample", "promote"), decision
    rv.update(reviewer=reviewer, reviewed_at=_now(), notes=notes,
              status={"reject": "rejected", "accept_sample": "accepted_sample", "promote": "promoted"}[decision])
    if decision == "promote":
        labels = json.loads((d / "labels.json").read_text())
        meta = json.loads((d / "scene.json").read_text())
        scaffold = _scaffold_template(proposal_id, labels,
                                      {"width_m": meta["width_m"], "soil_depth_m": meta["soil_depth_m"], "host": meta["host"]})
        rv["promotion"] = {"template_scaffold": str(scaffold)}
        print(f"PROMOTED {proposal_id} -> {scaffold}  (finalize ranges, then it joins Tier T)")
    elif decision == "accept_sample":
        rv["promotion"] = {"corpus_status": "approved_for_gprmax (origin=freeform_reviewed); run a batch on it"}
        print(f"ACCEPTED-AS-SAMPLE {proposal_id} (queued for a gprMax run into the corpus)")
    else:
        rej = PROPOSALS / "_rejected"; rej.mkdir(exist_ok=True)
        print(f"REJECTED {proposal_id} ({notes})")
    (d / "review.json").write_text(json.dumps(rv, indent=2), encoding="utf-8")


def list_proposals() -> None:
    mf = PROPOSALS / "proposals_manifest.jsonl"
    if not mf.exists():
        print("no proposals yet"); return
    for line in mf.read_text().splitlines():
        r = json.loads(line)
        d = PROPOSALS / r["proposal_id"]
        st = json.loads((d / "review.json").read_text())["status"] if (d / "review.json").exists() else r["status"]
        print(f"  {r['proposal_id']:28} {st:16} gateA={r['gate_a_ok']!s:5} objs={r['n_objects']}  \"{r['description'][:50]}\"")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("registry")
    pp = sub.add_parser("propose"); pp.add_argument("description"); pp.add_argument("--seed", type=int, default=0)
    sub.add_parser("list")
    rv = sub.add_parser("review"); rv.add_argument("proposal_id"); rv.add_argument("decision")
    rv.add_argument("--notes", default=""); rv.add_argument("--by", default="human")
    a = ap.parse_args()
    if a.cmd == "registry": print("wrote", write_registry())
    elif a.cmd == "propose": propose_freeform(a.description, seed=a.seed)
    elif a.cmd == "list": list_proposals()
    elif a.cmd == "review": review(a.proposal_id, a.decision, reviewer=a.by, notes=a.notes)


if __name__ == "__main__":
    main()

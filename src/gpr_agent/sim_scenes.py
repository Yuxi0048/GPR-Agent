"""Parametric FDTD GPR scenes: build123d CAD -> voxelize -> gprMax HDF5 -> B-scan (GPU).

The "hdf5 one": author the subsurface parametrically with **build123d** (OpenCASCADE
CAD, the text-to-cad engine), voxelize the solids onto the FDTD grid via OCP
point-in-solid, write the gprMax geometry package (`geometry.h5` `/data` index volume +
`materials.txt`), and run gprMax through `#geometry_objects_read` on the GPU. EM
properties (eps_r/sigma) come from the canonical KB `gpr_kb.reference` -- one source.
Known target materials => the polarity->material mapping can be validated against truth.

Simulator DEV scenes -- NOT TU1208 (the frozen test). Needs build123d + gprMax + a GPU.
Coordinates (metres): x horizontal across the survey; y vertical with 0 at the domain
bottom and the soil surface at `soil_depth_m`; thin in z (a 2-D gprMax model).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _gprmax_gpu_runner(out_dir: Path, gpu: bool):
    """Solver command + environment from GPR-Sim's runner (selects a VS-compatible
    CUDA 12.x toolkit -- a raw `python -m gprMax -gpu` defaults to CUDA v10.0 and
    fails to compile the kernel under a modern Visual Studio)."""
    from subsurface_platform.sim_engine.gprmax.runner import (
        GprMaxRunnerConfig, build_gprmax_runtime_environment,
        default_gpu_command_args, discover_default_solver_command,
    )
    env, _info = build_gprmax_runtime_environment(
        output_dir=out_dir, runner_config=GprMaxRunnerConfig(use_gpu=gpu, allow_cpu_fallback=True))
    solver = discover_default_solver_command()
    gpu_args = default_gpu_command_args(use_gpu=gpu)
    return solver, gpu_args, {**os.environ, **{k: str(v) for k, v in env.items()}}


# --------------------------------------------------------------------------- #
# EM properties -- one source (KB). Returns a gprMax #material spec.
# --------------------------------------------------------------------------- #
def em_spec(material: str) -> dict:
    """(eps_r, sigma, mu_r, sigma_star) for a material id. Metal => sigma='inf' (PEC-like)."""
    m = material.strip().lower()
    if m in ("metal", "pec", "steel"):
        return {"eps_r": 1.0, "sigma": "inf", "mu_r": 1.0, "sigma_star": 0.0}
    if m in ("air", "void", "free_space"):
        return {"eps_r": 1.0, "sigma": 0.0, "mu_r": 1.0, "sigma_star": 0.0}
    if m in ("water", "water_filled"):
        return {"eps_r": 81.0, "sigma": 0.05, "mu_r": 1.0, "sigma_star": 0.0}
    from gpr_kb import reference as ref
    if m in ref.load_materials():                          # host soils/rock from KB MATERIAL_DB
        mm = ref.get_material(m)
        sigma = 0.5 * (mm["sigma_min_mS_m"] + mm["sigma_max_mS_m"]) * 1e-3
        return {"eps_r": round(ref.material_eps_r_mid(m), 3), "sigma": round(sigma, 5), "mu_r": 1.0, "sigma_star": 0.0}
    if m in ("pvc", "hdpe"):                               # dielectric pipe wall from KB OBJECT_DB
        o = ref.get_object(f"{m}_pipe_empty")
        return {"eps_r": round(0.5 * (o["eps_r_surface_min"] + o["eps_r_surface_max"]), 3),
                "sigma": 0.0, "mu_r": 1.0, "sigma_star": 0.0}
    # Common construction / natural dielectrics the KB material DB does not carry yet.
    # Representative mid-range (eps_r, sigma S/m) at GPR frequencies (Daniels 2004;
    # Cassidy 2009; Annan 2005). SIMULATION inputs (physical properties, not GT-fit);
    # migrate into gpr_kb.reference when the KB gains construction materials.
    _LIT = {
        "concrete": (6.5, 0.02), "concrete_dry": (5.5, 0.01), "concrete_moist": (8.5, 0.04),
        "gravel": (5.0, 0.001), "wet_clay": (22.0, 0.05), "wet_sand": (22.0, 0.01),
        "silt": (12.0, 0.02), "loam": (12.0, 0.02), "asphalt": (5.0, 0.001),
        "brick": (4.0, 0.005), "wood": (6.0, 0.002), "root": (15.0, 0.012),
        "limestone": (7.0, 0.001), "granite": (5.5, 0.0005),
    }
    if m in _LIT:
        eps, sig = _LIT[m]
        return {"eps_r": eps, "sigma": sig, "mu_r": 1.0, "sigma_star": 0.0}
    raise ValueError(f"unknown material {material!r}")


def _fmt(v) -> str:
    return "inf" if str(v).lower() == "inf" else f"{float(v):.6g}"


# --------------------------------------------------------------------------- #
# Parametric CAD scene (build123d)
# --------------------------------------------------------------------------- #
@dataclass
class _Obj:
    solid: object        # build123d Solid
    material: str        # material id (-> em_spec)
    name: str


@dataclass
class Scene:
    width_m: float = 0.80
    soil_depth_m: float = 0.50
    air_gap_m: float = 0.08
    dx_m: float = 0.003
    soil_material: str = "moist_limestone"
    objects: list = field(default_factory=list)   # painted in order; later overrides earlier

    def _solid_box(self, x0, y0, x1, y1):
        from build123d import Box, Pos
        w, h, t = (x1 - x0), (y1 - y0), 4 * self.dx_m
        return Pos((x0 + x1) / 2, (y0 + y1) / 2, 0) * Box(w, h, t)

    def add_pipe(self, *, center_x_m: float, depth_m: float, radius_m: float, material: str):
        """A cylinder (axis out of the cross-section) at (x, surface-depth)."""
        from build123d import Cylinder, Pos
        cy = self.soil_depth_m - depth_m
        solid = Pos(center_x_m, cy, 0) * Cylinder(radius_m, 4 * self.dx_m)   # axis = z (thin dim)
        self.objects.append(_Obj(solid, material, f"{material}_pipe"))
        return self

    def add_void(self, *, center_x_m: float, depth_m: float, radius_m: float, material: str = "air"):
        return self.add_pipe(center_x_m=center_x_m, depth_m=depth_m, radius_m=radius_m, material=material)

    def add_layer(self, *, depth_top_m: float, thickness_m: float, material: str):
        y1 = self.soil_depth_m - depth_top_m
        self.objects.append(_Obj(self._solid_box(0, y1 - thickness_m, self.width_m, y1), material, f"{material}_layer"))
        return self

    def add_box(self, *, x_min_m: float, x_max_m: float, depth_top_m: float,
                depth_bottom_m: float, material: str, name: str | None = None):
        """A localized axis-aligned box (duct-bank envelope, trench backfill, slab/cap)."""
        y_top = self.soil_depth_m - depth_top_m          # shallower -> larger y
        y_bot = self.soil_depth_m - depth_bottom_m       # deeper    -> smaller y
        self.objects.append(_Obj(self._solid_box(x_min_m, y_bot, x_max_m, y_top),
                                 material, name or f"{material}_box"))
        return self

    # ---- voxelize via OCP point-in-solid ----
    def _classifier(self, solid):
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        c = BRepClass3d_SolidClassifier(solid.wrapped)
        return c

    def voxelize(self):
        """Return (volume int32 (nx, ny, 1), material_order list[str]). Index 0 = soil background."""
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON
        nx = int(round(self.width_m / self.dx_m))
        ny = int(round((self.soil_depth_m + self.air_gap_m) / self.dx_m))
        surf_j = int(round(self.soil_depth_m / self.dx_m))

        # material registry: 0=air background (above surface), 1=soil, then objects
        order = ["air", self.soil_material]
        idx_of = {"air": 0, self.soil_material: 1}
        vol = np.zeros((nx, ny, 1), dtype=np.int32)
        vol[:, :surf_j, 0] = 1                              # soil fills below the surface

        for ob in self.objects:                            # paint objects (later overrides)
            if ob.material not in idx_of:
                idx_of[ob.material] = len(order); order.append(ob.material)
            mi = idx_of[ob.material]
            cls = self._classifier(ob.solid)
            (x0, y0, _), (x1, y1, _) = ob.solid.bounding_box().min.to_tuple(), ob.solid.bounding_box().max.to_tuple()
            i0, i1 = max(0, int(x0 / self.dx_m) - 1), min(nx, int(x1 / self.dx_m) + 2)
            j0, j1 = max(0, int(y0 / self.dx_m) - 1), min(ny, int(y1 / self.dx_m) + 2)
            for i in range(i0, i1):
                xc = (i + 0.5) * self.dx_m
                for j in range(j0, j1):
                    yc = (j + 0.5) * self.dx_m
                    cls.Perform(gp_Pnt(xc, yc, 0.0), 1e-9)
                    if cls.State() in (TopAbs_IN, TopAbs_ON):
                        vol[i, j, 0] = mi
        return vol, order

    # ---- write gprMax geometry package ----
    def write_gprmax(self, out_dir: Path):
        import h5py
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        vol, order = self.voxelize()
        geo = out_dir / "geometry.h5"
        with h5py.File(geo, "w") as f:
            f.attrs["dx_dy_dz"] = (self.dx_m, self.dx_m, self.dx_m)
            f.create_dataset("/data", data=vol)            # material indices, 0..len(order)-1
        mats = out_dir / "materials.txt"
        lines = []
        for name in order:                                 # line order == index
            s = em_spec(name)
            lines.append(f"#material: {_fmt(s['eps_r'])} {_fmt(s['sigma'])} {_fmt(s['mu_r'])} {_fmt(s['sigma_star'])} {name}")
        mats.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out_dir / "scene.json").write_text(json.dumps({
            "width_m": self.width_m, "soil_depth_m": self.soil_depth_m, "dx_m": self.dx_m,
            "soil_material": self.soil_material, "material_order": order,
            "objects": [{"name": o.name, "material": o.material} for o in self.objects]}, indent=2))
        return geo, mats, order


# --------------------------------------------------------------------------- #
# gprMax deck + GPU run + B-scan
# --------------------------------------------------------------------------- #
def _deck_single(scene: Scene, geo: Path, mats: Path, *, fc_hz: float, source_x: float,
                 tx_rx_offset_m: float, time_window_s: float) -> str:
    """One source/rx position (no #src_steps) -> a single gprMax model.

    A B-scan is N of these run as SEPARATE processes -- `gprMax -n N -gpu` hits a
    pycuda multi-model context-cleanup crash, so we step the antenna in Python (the
    GPR-Sim approach) and the CUDA kernel cache keeps runs after the first fast.
    """
    height = scene.soil_depth_m + scene.air_gap_m
    ant_y = scene.soil_depth_m + 0.5 * scene.air_gap_m
    dx = scene.dx_m
    return "\n".join([
        "#title: build123d FDTD GPR trace",
        f"#domain: {scene.width_m:.3f} {height:.3f} {dx:.3f}",
        f"#dx_dy_dz: {dx:.3f} {dx:.3f} {dx:.3f}",
        f"#time_window: {time_window_s:.3g}",
        "",
        f"#waveform: ricker 1 {fc_hz:.6g} src_wave",
        f"#hertzian_dipole: z {source_x:.3f} {ant_y:.3f} 0 src_wave",
        f"#rx: {source_x + tx_rx_offset_m:.3f} {ant_y:.3f} 0",
        "",
        f"#geometry_objects_read: 0 0 0 {Path(geo).as_posix()} {Path(mats).as_posix()}",
        "",
    ]) + "\n"


def run_bscan(scene: Scene, *, out_dir: Path, fc_hz: float = 1.0e9, n_traces: int = 60,
              scan_start_m: float = 0.10, scan_step_m: float = 0.01, tx_rx_offset_m: float = 0.04,
              time_window_s: float = 1.2e-8, gpu: bool = True) -> dict:
    import h5py
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    geo, mats, order = scene.write_gprmax(out_dir)
    solver, gpu_args, env = _gprmax_gpu_runner(out_dir, gpu)
    traces, dt_s = [], None
    for k in range(n_traces):                              # one separate single-model run per trace
        sx = scan_start_m + k * scan_step_m
        deck = out_dir / f"t{k:03d}.in"
        deck.write_text(_deck_single(scene, geo, mats, fc_hz=fc_hz, source_x=sx,
                                     tx_rx_offset_m=tx_rx_offset_m, time_window_s=time_window_s), encoding="utf-8")
        cmd = list(solver) + [str(deck)] + (gpu_args if gpu else [])
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(out_dir), timeout=1200, env=env)
        out = out_dir / f"t{k:03d}.out"
        if not out.exists():
            raise RuntimeError(f"trace {k} produced no .out (rc={proc.returncode})\nstderr:\n{proc.stderr[-800:]}")
        with h5py.File(out, "r") as f:
            dt_s = float(f.attrs["dt"]); traces.append(np.asarray(f["rxs"]["rx1"]["Ez"], float))
    bscan = np.column_stack(traces)                        # (n_samples, n_traces)
    np.save(out_dir / "bscan.npy", bscan)
    return {"bscan": bscan, "dt_ns": dt_s * 1e9, "dx_m": scan_step_m, "material_order": order,
            "n_traces": n_traces, "out_dir": str(out_dir)}


# --------------------------------------------------------------------------- #
# CLI: one material-bearing scene (pipe of the given material in soil) -> B-scan.
#   PYTHONPATH=<GPR-Sim>/src;<GPR-KnowledgeBase>  python sim_scenes.py [material] [n_traces]
# materials: metal | air/void | water | pvc | hdpe
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time
    mat = sys.argv[1] if len(sys.argv) > 1 else "metal"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    s = Scene(width_m=0.8, soil_depth_m=0.5, dx_m=0.004, soil_material="moist_limestone")
    s.add_pipe(center_x_m=0.40, depth_m=0.15, radius_m=0.03, material=mat)
    t = time.perf_counter()
    r = run_bscan(s, out_dir=f"e:/github/_gprtools_tmp/cad_{mat}", fc_hz=1.0e9, n_traces=n,
                  scan_start_m=0.10, scan_step_m=0.02, time_window_s=1.2e-8, gpu=True)
    print(f"{mat}: B-scan {r['bscan'].shape} in {time.perf_counter()-t:.0f}s | dt={r['dt_ns']:.4f} ns | "
          f"materials={r['material_order']} | -> {r['out_dir']}")

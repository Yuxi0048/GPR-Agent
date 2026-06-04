"""Synthetic DEV scenes for leakage-free VLM refinement (NOT a benchmark).

Plants a few hyperbolas (proper moveout) + a shallow horizontal layer (clutter, like
the TU1208 band) + noise. These are a DEV set you may iterate prompts on -- the frozen
TU1208/Purdue test set is never used for refinement. The planted positions are a
DEV-ONLY sanity signal; the primary refinement signals are the PROCESS rubric and
agreement with the deterministic oracle (both GT-free). For richer scenes, swap this
for GPR-Sim (subsurface_platform).
"""
from __future__ import annotations

import numpy as np


def _ricker(n: int = 31, fc_samples: float = 6.0) -> np.ndarray:
    t = np.arange(-(n // 2), n - n // 2)
    a = (1 - 2 * (np.pi * t / fc_samples) ** 2) * np.exp(-((np.pi * t / fc_samples) ** 2))
    return a


def synth_scene(seed: int = 0, *, n_s: int = 320, n_t: int = 240, dt_ns: float = 0.4,
                dx_m: float = 0.04, v: float = 0.1, k: int = 4, noise: float = 0.03):
    """Return (image (n_s, n_t), dt_ns, dx_m, planted[(x_m, depth_m), ...])."""
    rng = np.random.default_rng(seed)
    img = noise * rng.standard_normal((n_s, n_t))
    wav = _ricker(31)
    img[int(rng.integers(15, 30)), :] += 0.5            # shallow horizontal clutter band
    planted = []
    for _ in range(k):
        x0 = float(rng.uniform(0.15, 0.85) * n_t)
        t0 = float(rng.uniform(0.30, 0.85) * n_s)        # apex sample
        amp = float(rng.uniform(0.6, 1.0))
        t0_ns = t0 * dt_ns
        for t in range(n_t):
            dx = (t - x0) * dx_m                          # offset (m)
            s = np.sqrt(t0_ns ** 2 + (2 * dx / v) ** 2) / dt_ns   # hyperbola sample
            si = int(round(s))
            a = amp * np.exp(-((abs(t - x0) / (0.25 * n_t)) ** 2))
            lo = si - len(wav) // 2
            for j, wv in enumerate(wav):
                ki = lo + j
                if 0 <= ki < n_s:
                    img[ki, t] += a * wv
        planted.append((round(x0 * dx_m, 2), round(0.5 * v * t0_ns, 2)))
    return np.ascontiguousarray(img), dt_ns, dx_m, sorted(planted, key=lambda p: p[1])

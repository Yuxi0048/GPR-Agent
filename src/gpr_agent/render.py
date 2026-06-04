"""Render a Session array (B-scan / migrated section) to a PNG the VLM can see.

The VLMs (A2/A3) never receive arrays -- they receive an `ImageRef` handle plus the
PNG rendered here (stored in `Session.renders[handle]`). Labelled axes (x in metres,
depth in metres at the nominal/uncalibrated velocity) so the model can read positions
without us leaking the raw pixels into the prompt. matplotlib Agg, no display.
"""
from __future__ import annotations

import io

import numpy as np

from .contracts import ImageRef


def render_section(array: np.ndarray, *, dt_ns: float | None, dx_m: float | None,
                   velocity: float = 0.1, title: str = "", dpi: int = 110) -> bytes:
    """(n_rows, n_traces) energy image -> labelled grayscale PNG bytes (x[m] vs depth[m])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.asarray(array, float)
    n_r, n_t = img.shape
    dt_ns = dt_ns or 1.0
    dx_m = dx_m or 1.0
    depth_max = 0.5 * velocity * (n_r - 1) * dt_ns
    x_max = n_t * dx_m

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.imshow(img, aspect="auto", cmap="gray_r", vmin=0, vmax=float(np.percentile(img, 99)),
              extent=[0, x_max, depth_max, 0])
    ax.set_xlabel("x (m)")
    ax.set_ylabel(f"depth (m, uncalibrated v={velocity:g} m/ns)")
    if title:
        ax.set_title(title)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def into_session(session, img: ImageRef, *, velocity: float = 0.1, title: str = "") -> bytes:
    """Render the array at `img.handle` and store the PNG in `session.renders[img.handle]`."""
    png = render_section(session.arrays[img.handle], dt_ns=img.dt_ns, dx_m=img.dx_m,
                         velocity=velocity, title=title or img.domain)
    session.renders[img.handle] = png
    return png

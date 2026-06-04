"""Session -- the Python-side state the agents share (arrays + renders).

Lives in its own module (no pydantic-ai, no numpy import at module load) so both the
VLM path (`agents.py`) and the keyless engine (`deterministic.py`) use ONE Session
type. The arrays the VLMs must never see live here; messages carry only `ImageRef`
handles into `arrays`, and `renders` holds PNGs for the VLM by the same handle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    arrays: dict[str, Any] = field(default_factory=dict)     # handle -> np.ndarray
    renders: dict[str, bytes] = field(default_factory=dict)  # handle -> PNG (for the VLM)
    images: dict[str, Any] = field(default_factory=dict)     # handle -> ImageRef (carries dt_ns/dx_m)

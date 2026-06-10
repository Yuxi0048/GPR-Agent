"""Inter-agent contracts (the careful part) — plain pydantic v2.

These are the typed handoffs the agents + prompts reference. **Directional by
design** (this is what structurally blocks circular reasoning):
  - A4 emits `Task` / `Query` **down** to workers — never a conclusion;
  - workers emit `DomainEvidence` **up** to A4;
  - workers never read each other or A4 — fusion happens only at A4.

The **reasoning** types (`Domain`, `Region`, `ToolCall`, `Provenance`, `Detection`,
`DomainEvidence`, `Query`, `HypothesisStatus`, `Hypothesis`, `InterpretationCase`) now live in the
`gpr_reasoning` library (GPR-Reasoning) and are **re-exported here for back-compat**; the
**imaging / orchestration** contracts (`ImageRef`, `ImagingResult`, `Task`) stay agent-local.
Dependency direction: GPR-Agent → `gpr_reasoning` (reasoning is a foundation, never the reverse).

DEV scaffold — design substrate for the 4-agent system (see ../AGENTS_DESIGN.md).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Reasoning model (single source: gpr_reasoning, GPR-Reasoning) — re-exported for back-compat.
from gpr_reasoning import (
    Detection,
    Domain,
    DomainEvidence,
    Hypothesis,
    HypothesisStatus,
    InterpretationCase,
    Provenance,
    Query,
    Region,
    ToolCall,
)


# --------------------------------------------------------------------------- #
# A1 imaging output — by REFERENCE (arrays stay in the session, never in messages)
# (imaging artifacts are agent-local, not reasoning)
# --------------------------------------------------------------------------- #
class ImageRef(BaseModel):
    domain: Domain
    handle: str                                 # key into the Session's array store
    n_samples: int
    n_traces: int
    dt_ns: Optional[float] = None
    dx_m: Optional[float] = None


class ImagingResult(BaseModel):
    bscan: ImageRef
    subsurface: Optional[ImageRef] = None       # migrated image (if A1 migrated)
    velocity_m_per_ns: Optional[float] = None
    velocity_calibrated: bool = False           # honesty flag — see review (calibration)
    provenance: Provenance


# --------------------------------------------------------------------------- #
# A4 dispatch — DOWN (a Task, never a conclusion). (orchestration, agent-local)
# --------------------------------------------------------------------------- #
class Task(BaseModel):
    """A4 -> worker: analyze a domain image. Carries NO findings (anti-circular)."""
    kind: Literal["analyze"] = "analyze"
    domain: Domain
    image: ImageRef
    focus_region: Optional[Region] = None


__all__ = [
    # reasoning (from gpr_reasoning)
    "Domain", "Region", "ToolCall", "Provenance", "Detection", "DomainEvidence",
    "Query", "HypothesisStatus", "Hypothesis", "InterpretationCase",
    # imaging / orchestration (agent-local)
    "ImageRef", "ImagingResult", "Task",
]

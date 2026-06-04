"""Inter-agent contracts (the careful part) — plain pydantic v2.

These are the typed handoffs the agents + prompts reference. **Directional by
design** (this is what structurally blocks circular reasoning):
  - A4 emits `Task` / `Query` **down** to workers — never a conclusion;
  - workers emit `DomainEvidence` **up** to A4;
  - workers never read each other or A4 — fusion happens only at A4.

DEV scaffold — design substrate for the 4-agent system (see ../AGENTS_DESIGN.md),
not a formal/finished schema. Runs on pydantic alone (no pydantic-ai needed).
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

Domain = Literal["radargram", "subsurface"]
Region = tuple[float, float, float, float]   # (x0, z0, x1, z1) in metres


# --------------------------------------------------------------------------- #
# Provenance (every artifact carries it -> evidence-based + auditable)
# --------------------------------------------------------------------------- #
class ToolCall(BaseModel):
    tool: str
    params: dict = Field(default_factory=dict)


class Provenance(BaseModel):
    agent: str                                  # which agent produced this
    tools: list[ToolCall] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------- #
# A1 imaging output — by REFERENCE (arrays stay in the session, never in messages)
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
# Worker evidence — UP to A4 (A2 radargram / A3 subsurface)
# --------------------------------------------------------------------------- #
class Detection(BaseModel):
    """One detected feature = the unit of evidence."""
    id: str
    domain: Domain
    x_m: Optional[float] = None
    depth_m: Optional[float] = None
    twtt_ns: Optional[float] = None
    kind: str = "unknown"                       # apex / hyperbola / reflector / blob / ...
    confidence: float = Field(0.5, ge=0.0, le=1.0)   # RELATIVE rank, NOT a calibrated prob
    polarity: Optional[str] = None              # POSITIVE / NEGATIVE / AMBIGUOUS (A2 apex polarity)
    supporting_tools: list[str] = Field(default_factory=list)   # evidence-based (>=1 required)


class DomainEvidence(BaseModel):
    domain: Domain
    detections: list[Detection] = Field(default_factory=list)
    quality: str = ""                           # low SNR / ringing / ambiguous regions...
    provenance: Provenance


# --------------------------------------------------------------------------- #
# A4 dispatch — DOWN (a Task/Query, never a conclusion)
# --------------------------------------------------------------------------- #
class Task(BaseModel):
    """A4 -> worker: analyze a domain image. Carries NO findings (anti-circular)."""
    kind: Literal["analyze"] = "analyze"
    domain: Domain
    image: ImageRef
    focus_region: Optional[Region] = None


class Query(BaseModel):
    """A4 -> worker: re-examine a region. A falsifiable QUESTION, not a conclusion."""
    kind: Literal["re_examine"] = "re_examine"
    domain: Domain
    region: Region
    question: str                               # "is there an apex consistent with X here?"


# --------------------------------------------------------------------------- #
# A4 hypotheses + final interpretation — hypothesis-driven, evidence-based
# --------------------------------------------------------------------------- #
class HypothesisStatus(str, Enum):
    candidate = "candidate"
    supported = "supported"
    refuted = "refuted"
    ambiguous = "ambiguous"


class Hypothesis(BaseModel):
    id: str
    statement: str                              # "metal pipe at x~2.0 m, depth~0.8 m"
    status: HypothesisStatus = HypothesisStatus.candidate
    kind: str = "unknown"                       # object kind: pipe / void / rebar / layer / ...
    confidence: float = Field(0.0, ge=0.0, le=1.0)        # relative; uncalibrated
    x_m: Optional[float] = None                 # best location estimate (for scoring/downstream)
    depth_m: Optional[float] = None             # uncalibrated unless imaging.velocity_calibrated
    material: Optional[str] = None              # candidate material role (from polarity -> KB lookup)
    predicted_in: list[Domain] = Field(default_factory=list)   # signatures it predicts
    supporting_evidence: list[str] = Field(default_factory=list)  # Detection ids (>=1 to be 'supported')
    refuting_evidence: list[str] = Field(default_factory=list)
    cross_domain_confirmed: bool = False        # matched in BOTH domains? (de-dup did this)


class InterpretationCase(BaseModel):
    """A4 final output: evidence-based, confidence-graded, never 'confirmed utility'."""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    follow_up_queries: list[Query] = Field(default_factory=list)   # bounded re-plan
    overall_note: str = ""
    is_tentative: bool = True                   # overclaim guard (must stay True)
    provenance: Provenance

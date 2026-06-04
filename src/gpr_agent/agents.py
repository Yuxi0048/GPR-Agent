"""The 4-agent system in Pydantic-AI format (DEV scaffold / design substrate).

Topology (see ../AGENTS_DESIGN.md): A1 imaging -> (A2 radargram || A3 subsurface)
-> A4 leader. Each agent has a typed `output_type` (a contract) and a `deps_type`
(the Session holding arrays the VLM never sees). The directional contracts make
circular reasoning structurally hard: A4 emits Task/Query down, workers emit
DomainEvidence up, workers never see each other or A4.

Needs `pip install pydantic-ai`. The SYSTEM PROMPTS now live as reviewable markdown
in `prompts/` (DRAFT v0.1, grown from ../from_simulationv3/gpr_interpretation/llm/
prompts/*.md) and are loaded via `.prompts`. Run `demo_with_testmodel()` to exercise
the wiring with no API key (pydantic-ai's TestModel = the StubBackend pattern).
"""
from __future__ import annotations

import os
from typing import Any

from pydantic_ai import Agent, BinaryContent, RunContext

from . import prompts, render, tools
from .contracts import (
    DomainEvidence, ImageRef, ImagingResult, InterpretationCase, Provenance, Task, ToolCall,
)
from .deterministic import LoopResult, fuse
from .session import Session   # shared with the keyless engine (deterministic.py)

# Default model (vision-capable). Override per-run or via env GPR_AGENT_MODEL
# (e.g. "anthropic:claude-sonnet-4-6", "openai-chat:gpt-4o").
MODEL = os.environ.get("GPR_AGENT_MODEL", "anthropic:claude-sonnet-4-6")


# --------------------------------------------------------------------------- #
# A1 — imaging (tool-augmented): raw -> preprocessed B-scan (+ migrated subsurface)
# --------------------------------------------------------------------------- #
a1 = Agent(MODEL, deps_type=Session, output_type=ImagingResult, system_prompt=prompts.A1)


# --------------------------------------------------------------------------- #
# A2 — radargram analyzer/detector (tool-use VLM)
# --------------------------------------------------------------------------- #
a2 = Agent(MODEL, deps_type=Session, output_type=DomainEvidence, system_prompt=prompts.A2)


def _ref(ctx: RunContext[Session], handle: str, domain: str):
    """Resolve the full ImageRef (dt_ns/dx_m) for a handle; (None, err) if unknown."""
    arrs = ctx.deps.arrays
    if handle not in arrs:
        return None, {"error": f"unknown handle {handle!r}; available handles: {list(arrs)}"}
    img = ctx.deps.images.get(handle)
    if img is None:                                            # reconstruct (units may be in samples/traces)
        a = arrs[handle]
        img = ImageRef(domain=domain, handle=handle, n_samples=a.shape[0], n_traces=a.shape[1])
    return img, None


@a2.tool
def dip(ctx: RunContext[Session], handle: str) -> dict[str, Any]:
    """Structure-tensor dip on the B-scan -> a small numeric summary (not arrays)."""
    if handle not in ctx.deps.arrays:
        return {"error": f"unknown handle {handle!r}; available handles: {list(ctx.deps.arrays)}"}
    return tools.dip_summary(ctx.deps, handle)


@a2.tool
def detect_apices(ctx: RunContext[Session], handle: str, velocity_m_per_ns: float = tools.NOMINAL_V) -> dict[str, Any]:
    """Hyperbola-apex detection on the B-scan -> the radargram DomainEvidence (summary)."""
    img, err = _ref(ctx, handle, "radargram")
    return err if err else tools.analyze_radargram(ctx.deps, img, velocity=velocity_m_per_ns).model_dump()


@a2.tool
def check_polarity(ctx: RunContext[Session], signed_handle: str, sample_idx: int, trace_idx: int,
                   fc_mhz: float = 1000.0, dt_ns: float = 0.1) -> dict[str, Any]:
    """Apex polarity (POSITIVE/NEGATIVE/AMBIGUOUS) at (sample, trace) on the RAW B-scan.

    Use the raw/signed handle (not the envelope). NEGATIVE => target eps_r < host
    (air void / plastic) OR a conductor (metal); POSITIVE => higher eps_r (water).
    Pass to A4 with the host eps_r for the candidate-material lookup.
    """
    if signed_handle not in ctx.deps.arrays:
        return {"error": f"unknown handle {signed_handle!r}; available: {list(ctx.deps.arrays)}"}
    return tools.apex_polarity(ctx.deps, signed_handle, sample_idx, trace_idx,
                               fc_hz=fc_mhz * 1e6, dt_ns=dt_ns)


@a2.tool
def check_amplitude(ctx: RunContext[Session], signed_handle: str, sample_idx: int, trace_idx: int,
                    fc_mhz: float = 1000.0, dt_ns: float = 0.1) -> dict[str, Any]:
    """Apex amplitude SIGNATURE (relative SNR / rel-to-direct / ringing) -> strength +
    conductor_like, with a LOW-reliability flag. Amplitude is confounded by gain/depth/
    attenuation/size, so treat the result as a WEAK prior: only a confident conductor
    ringing or a clear 'strong' should sway the material/kind diagnosis."""
    if signed_handle not in ctx.deps.arrays:
        return {"error": f"unknown handle {signed_handle!r}; available: {list(ctx.deps.arrays)}"}
    return tools.apex_amplitude(ctx.deps, signed_handle, sample_idx, trace_idx,
                                fc_hz=fc_mhz * 1e6, dt_ns=dt_ns)


# (more A2 tools: fk_spectrum, radon, hyperbola_fit ... -> add to tools.py + cite them)


# --------------------------------------------------------------------------- #
# A3 — subsurface-model analyzer/detector (tool-use VLM)
# --------------------------------------------------------------------------- #
a3 = Agent(MODEL, deps_type=Session, output_type=DomainEvidence, system_prompt=prompts.A3)


@a3.tool
def detect_blobs(ctx: RunContext[Session], handle: str, velocity_m_per_ns: float = tools.NOMINAL_V) -> dict[str, Any]:
    """2-D blob-NMS on the migrated/subsurface image -> the subsurface DomainEvidence (summary)."""
    img, err = _ref(ctx, handle, "subsurface")
    return err if err else tools.analyze_subsurface(ctx.deps, img, velocity=velocity_m_per_ns).model_dump()


# --------------------------------------------------------------------------- #
# A4 — leader / reviewer / interpreter (hypothesis-driven, evidence-based)
# --------------------------------------------------------------------------- #
# `a4` is the full-interpretation agent (emits the whole InterpretationCase). It is
# error-prone when asked to re-emit a large de-duplicated skeleton (returns empty),
# so the current HYBRID uses deterministic geometry for the hypotheses and `a4_note`
# (below) for the VLM interpretation narrative -- robust, and DUP-001 holds by
# construction. Per-hypothesis VLM kind/material annotation is a future enrichment.
a4 = Agent(MODEL, deps_type=Session, output_type=InterpretationCase, system_prompt=prompts.A4)

_A4_NOTE_PROMPT = (
    "You are the GPR interpretation leader writing the summary note over a FIXED, "
    "de-duplicated set of hypotheses (you do not change them). Be evidence-based and "
    "honest: NO overclaim; confidences are RELATIVE, not calibrated; velocity is "
    "nominal/uncalibrated so depths are uncertain. Output ONLY the note text (no JSON)."
)
a4_note = Agent(MODEL, deps_type=Session, output_type=str, system_prompt=_A4_NOTE_PROMPT)


# --------------------------------------------------------------------------- #
# Orchestrator-workers harness: A1 (det. imaging) -> VLM A2 || A3 -> VLM A4
# --------------------------------------------------------------------------- #
def _png(session: Session, handle: str) -> BinaryContent:
    return BinaryContent(data=session.renders[handle], media_type="image/png")


def _a2_msg(img: ImageRef) -> str:
    return (f"Analyze this B-scan (radargram / time domain), stored at session handle "
            f"'{img.handle}' ({img.n_samples} samples x {img.n_traces} traces; dt={img.dt_ns} ns, "
            f"dx={img.dx_m} m). Call your tools on that handle to GROUND every detection, then "
            f"emit DomainEvidence(domain='radargram'). The attached image is the same B-scan "
            f"(darker = stronger envelope energy).")


def _a3_msg(img: ImageRef) -> str:
    return (f"Analyze this MIGRATED subsurface section (depth domain), handle '{img.handle}' "
            f"({img.n_samples} depth-rows x {img.n_traces} traces; dx={img.dx_m} m; depth uses an "
            f"UNCALIBRATED velocity). Call your tools on that handle, then emit "
            f"DomainEvidence(domain='subsurface'). The attached image is that section.")


def _a4_note_msg(fused: list, imaging: ImagingResult) -> str:
    confirmed = sum(h.cross_domain_confirmed for h in fused)
    locs = "; ".join(f"{h.id}:x~{h.x_m:.1f}m,z~{h.depth_m:.1f}m,{h.status.value}"
                     for h in fused[:15] if h.x_m is not None and h.depth_m is not None)
    return (f"{len(fused)} de-duplicated hypotheses from a deterministic cross-domain association "
            f"({confirmed} corroborated in both domains). Velocity {imaging.velocity_m_per_ns} m/ns, "
            f"calibrated={imaging.velocity_calibrated}. Write a concise interpretation NOTE (3-6 sentences): "
            f"the scene story, where the strongest targets are, the main uncertainties, and an explicit "
            f"reminder that confidences are RELATIVE and depths uncalibrated. Hypotheses (top): {locs}")


def orchestrate(session: Session, raw_handle: str, *, dt_ns: float, dx_m: float,
                velocity: float = tools.NOMINAL_V, migrate: str | bool = "adjoint",
                model: Any = None) -> LoopResult:
    """VLM loop: deterministic A1 imaging (+render) -> VLM A2/A3 (see image + call tools)
    -> VLM A4 fuse. Returns the same contracts as the deterministic oracle, so the two
    are directly comparable. A2/A3 run sequentially here (independent; gather in async)."""
    model = model or MODEL
    imaging = tools.image(session, raw_handle, dt_ns=dt_ns, dx_m=dx_m, velocity=velocity, migrate=migrate)
    render.into_session(session, imaging.bscan, velocity=velocity, title="B-scan (envelope)")
    render.into_session(session, imaging.subsurface, velocity=velocity, title="migrated subsurface section")

    with a2.override(model=model), a3.override(model=model):
        r_r = a2.run_sync([_a2_msg(imaging.bscan), _png(session, imaging.bscan.handle)], deps=session)
        r_s = a3.run_sync([_a3_msg(imaging.subsurface), _png(session, imaging.subsurface.handle)], deps=session)
    # Hybrid A4: deterministic geometry builds the de-duplicated hypotheses (DUP-001
    # holds by construction); the VLM writes only the interpretation narrative.
    fused_hyps, fused_queries = fuse(r_r.output, r_s.output, imaging)
    with a4_note.override(model=model):
        r_4 = a4_note.run_sync(_a4_note_msg(fused_hyps, imaging), deps=session)
    case = InterpretationCase(
        hypotheses=fused_hyps, follow_up_queries=fused_queries, overall_note=r_4.output,
        is_tentative=True,
        provenance=Provenance(agent="A4", tools=[ToolCall(tool="fuse")],
                              notes="hybrid: deterministic geometry (de-dup) + VLM narrative"))
    tok = {"input": 0, "output": 0, "requests": 0}
    for r in (r_r, r_s, r_4):
        u = r.usage
        tok["input"] += getattr(u, "input_tokens", 0) or 0
        tok["output"] += getattr(u, "output_tokens", 0) or 0
        tok["requests"] += getattr(u, "requests", 0) or 0
    return LoopResult(imaging=imaging, evidence=[r_r.output, r_s.output], case=case, rounds=0, usage=tok)


def demo_with_testmodel(n_samples: int = 64, n_traces: int = 64) -> LoopResult:
    """Exercise the VLM wiring with NO API key (TestModel auto-fills the typed outputs).

    TestModel only checks the *wiring* (types validate); it returns empty/zeroed outputs,
    not real evidence. For a keyless loop with REAL detections use
    `deterministic.orchestrate_deterministic` (shared tools, no LLM).
    """
    import numpy as np
    from pydantic_ai.models.test import TestModel

    s = Session(arrays={"raw": np.zeros((n_samples, n_traces))})
    return orchestrate(s, "raw", dt_ns=0.1, dx_m=0.02, migrate=False, model=TestModel())

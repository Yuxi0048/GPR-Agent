# Agent work — two generations, compared

There are **two codebases** for LLM/tool-augmented GPR interpretation, both by the
author. They are different *architectures of the same idea*, not line-diffable —
so this is a capability/maturity comparison and a consolidation recommendation.

| | **`temp_src/gpr_agent_*`** (workbench prototype) | **`from_simulationv3/gpr_interpretation/`** (full framework) |
|---|---|---|
| Size | ~750 lines (`gpr_agent_tools` 613 + `sandbox` 137) | ~11,000 lines (44 files) |
| mtime | 2026-06-01 (slightly newer) | 2026-05-31 |
| Architecture | One **tool layer** + a **deterministic scripted harness** (no live LLM) | **Multi-agent framework**: perception (deterministic + tool-use) → hypothesis → grounding → reporter, typed Pydantic handoffs, replayable/auditable |
| Tools | 7 (COMPUTE: envelope/dip/blob/hyperbola; RENDER: overlay/crop/bbox; FUSION: `classify_event`) | **40** (`agents/perception_tools.py`, gpr_data_processing ops as Anthropic-API tools) |
| LLM | none (scripted plan stands in) | `LLMBackend` Protocol + Stub / ClaudeCodeSubagent / Anthropic (live) backends |
| Contracts | ad-hoc `ToolResult` dataclass | pydantic `contracts/` (RadargramFeatureSet, SubsurfaceHypothesis, GroundingResult, InterpretationCase, Provenance) |
| UI | none | `gpr_agent_ui.py` (1,544) + `ui/state.py` (683) — browser workbench |
| Entry point | `gpr_agent_sandbox.py __main__` (TU1208 demo) | `pipelines.single_radargram.run_interpretation` + `cli.py` |
| Extras | palette-flip ambiguity probe; seismic-palette overlay | `training/` (n2v/dncnn), `inference/remote_hyperbola_client`, `convergence`, `public_surface_registry()`, tests |
| Deps | `gpr_data_processing.*`, **stale `visualization`** (→ gpr_viz), a `run_*.py` | `gpr_data_processing.*` (+ `fitting/agent_contracts`), `pydantic`, `anthropic` |

## Relationship
The framework (`gpr_interpretation`) is the **mature, canonical** system — typed,
multi-agent, LLM-backed, UI'd, tested, with a stable public surface. The
`temp_src` pair is a **lean re-prototype of just the perception/tool layer**
(its 7 tools are a subset of the framework's 40), exploring two distinctive
ideas worth keeping: the **palette-flip ambiguity probe** and the **no-LLM
deterministic sandbox harness** (great for testing the tool loop without an API).

## Recommendation
1. **Base = `gpr_interpretation`** (the framework). Keep it as the agent system.
2. **Fold in from the prototype:** the palette-flip probe + `classify_event`
   fusion heuristic (if not already covered) and the deterministic sandbox as a
   **test harness** for the tool-use loop.
3. **Fix the consolidation seams:** `gpr_interpretation` imports
   `gpr_data_processing.fitting.agent_contracts`, which lives in the *legacy*
   GPR-Simulationv3 tree, **not** in GPR-Tools — so `agent_contracts` must be
   promoted into GPR-Tools (or vendored into the agent) before this runs against
   the split repos. Same for any other legacy-only `gpr_data_processing` pieces.
4. **Home (decision):** this is a coherent ~11 k-line subsystem with its own
   contracts + tests + public surface — large and distinct enough to justify
   **its own repo (`GPR-Agent` / `GPR-Reasoning`, an L2/L3 app on top of
   GPR-Tools)** — *or* it lives in the Workbench as the interpretation app
   (`apps/agent`). Unlike the active-learning scripts, this one is repo-worthy.

## Status
Copied here for review only. Nothing deleted; the `temp_src` prototype is still
in place. Decide the home + the agent_contracts promotion, then consolidate.

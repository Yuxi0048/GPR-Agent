# Changelog

All notable changes to GPR-Agent are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/); this project uses SemVer.

## [0.1.0] — 2026-06-04
Initial release — **promoted out of the GPR-Workbench `experiments/agent` incubator**
(the winning `design_pydantic_ai` draft), preserved as the baseline that demonstrates
NL-instructed simulation + the keyless agentic workflow.

### Added
- **4-agent topology** — A1 imaging → (A2 radargram ‖ A3 subsurface) → A4 leader, with
  directional pydantic contracts (`Task`/`Query` down, `DomainEvidence` up).
- **Keyless deterministic oracle** (`deterministic.py`) — the whole topology with no LLM:
  a gated bipartite geometric fuse (de-dup + falsification + bounded re-plan). FROZEN and
  pin-tested (`test_oracle_frozen.py`) so it stays a stable reference.
- **A4 control loop** (`control.py`) — diagnoses upstream defocus via a GT-free focus
  metric and acts (autofocus re-migrate → re-run A2/A3/A4).
- **Perception tools** (`tools.py`) — imaging (BGR + Kirchhoff migration), 1-D apex /
  polarity / amplitude, 2-D blob NMS, dip — over GPR-Tools.
- **Material/kind diagnosis** (`materials.py`) — polarity (+ optional amplitude/conductor
  cue, do-no-harm gated) → candidate material/kind, read from the canonical GPR-KB
  reference. Honest: *narrows*, never a unique ID.
- **VLM agents** (`agents.py`, optional `[vlm]`) — the 4 agents in Pydantic-AI; hybrid A4.
- **Review framework** (`review.py`, `review_llm.py`) — 9 deterministic checks + a
  controlled LLM judge, separate from the agent under review.
- **NL-instructed simulation** (`sim_scenes.py`, optional `[sim]`) — parametric scene →
  build123d CAD → voxelize → gprMax geometry HDF5 → FDTD B-scan (GPU via GPR-Sim runtime).
- Boundary test, `pyproject`, packaging.

### Principle
- **No calibration** documented as the first principle; confidences are relative ranks,
  GT evaluates and never tunes.

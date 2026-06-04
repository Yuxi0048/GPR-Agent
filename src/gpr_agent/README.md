# gpr_agent — the 4-agent system in Pydantic-AI format (DEV draft)

Design substrate for the agent in [`../AGENTS_DESIGN.md`](../AGENTS_DESIGN.md),
expressed in [Pydantic-AI](https://ai.pydantic.dev). **Not a formal experiment** —
the prompts are placeholders and the tools are slots; this is where the *careful*
contract + prompt design happens.

## Files
| File | What | Runs without an API key? |
|---|---|---|
| `contracts.py` | typed inter-agent handoffs (`Task`/`Query` down, `DomainEvidence` up, `Hypothesis`, `InterpretationCase`, `Provenance`) | ✅ pydantic only |
| `session.py` | the shared `Session` (arrays/renders the VLMs never see) | ✅ stdlib only |
| `prompts/` | the **4 system prompts** as reviewable markdown (`a1_imaging` · `a2_radargram` · `a3_subsurface` · `a4_leader`, DRAFT v0.1) + a no-dep `load()` | ✅ stdlib only |
| `tools.py` | **deterministic perception backbone** — A1 `image` (BGR + real Kirchhoff/LSM **migration** → subsurface section), A2 `analyze_radargram` (1-D apex profile), A3 `analyze_subsurface` (2-D blob NMS on the migrated section), `dip_summary`; shared by the engine AND the VLM tool slots | ✅ numpy/scipy + gpr-data-processing (migration needs `[migration]`) |
| `deterministic.py` | the **keyless engine** — `orchestrate_deterministic` (A1→A2‖A3→A4) + `fuse` (cross-domain association → de-dup → falsification → bounded re-plan) → real `InterpretationCase`, NO LLM | ✅ numpy + gpr-data-processing |
| `agents.py` | the 4 VLM agents (prompts from `prompts/`, tool slots delegating to `tools.py`) + `orchestrate()` + `demo_with_testmodel()` | needs `pip install pydantic-ai`; demo uses `TestModel` |
| `review.py` | the **review framework** — question bank + deterministic checks over the I/O | ✅ pydantic only |
| `test_prompts.py` · `test_deterministic.py` | guards: rubric ↔ prompt linkage; and the full keyless loop on synthetic data (evidence-cited, relative confidence, de-dup, rubric passes) | ✅ stdlib / +engine |

**Run the keyless loop, scored vs GPR-Bench + reviewed** (the deterministic oracle —
real detections, no LLM, no key):
```
PYTHONPATH=<GPR-Bench>;<GPR-Tools>/src  python ../run_deterministic_loop.py   # tu1208
```
On TU1208 the cross-domain "confirmed only" set out-precisions the A2-only baseline
(de-dup filters false apices); all 6 deterministic review checks pass. *Dev sanity
bar, not a formal benchmark.*

## Why Pydantic-AI here
- The **contracts are the output_type** — structured outputs are validated, so a
  handoff is either a valid `DomainEvidence`/`InterpretationCase` or it fails loudly.
- **`TestModel` / `FunctionModel`** = the StubBackend pattern: run/CI the whole
  loop with no key (the demo), then swap in `anthropic:...` for real reasoning.
- Model-agnostic + dependency injection (`deps_type=Session`) keeps the arrays out
  of the prompt (the VLM sees only `ImageRef`s + tool summaries).

## The review framework (extend this)
`review.py` turns the A4 rubric into **review questions** over each artifact, with
check types `deterministic | llm_judge | human`. To add your own instructions/questions:
1. append a `ReviewQuestion(id, dimension, target, question, check)` to `REVIEW_BANK`;
2. if auto-checkable, write `def _check(a: Artifacts) -> (bool, str)` and register it in `CHECKERS[<id>]`;
3. `run_reviews(Artifacts(imaging=, evidence=[...], case=))` runs the deterministic ones and lists the rest as prompts; `summary(...)` tallies them.

Dimensions = consistency · physical_plausibility · calibrated_uncertainty ·
no_circular_reasoning · no_double_count · no_overclaim · evidence_based. Targets =
`A1.out` · `A2.out` · `A3.out` · `A4.out` · `A4.query`.

## Still to design (carefully)
- **Refine the system prompts** — `prompts/*.md` are DRAFT v0.1. Each ends with
  **open design questions** to resolve (re-plan budget, disagreement policy,
  confidence scale, velocity ownership, id scheme). Edit the markdown directly; the
  `test_prompts.py` markers keep the non-negotiable discipline from being dropped.
- **Velocity calibration** — A1 uses a nominal velocity (`velocity_calibrated=False`);
  wire `migration.autofocus_velocity` so a focus sweep *earns* calibration (and
  honest depths). Until then confidences/depths stay relative.
- **Residual band + LSM** — even post-migration a faint ~0.65 m band lingers
  (`adjoint` mode); try `migrate="lsm"` (least-squares) / PnP, or an A3 lateral-
  continuity filter, to clean the remaining ambiguous blobs.
- **The VLM path** — render `ImageRef`→PNG into `Session.renders` (gpr-viz), then
  swap `FunctionModel`/`anthropic:` into A2/A3/A4 and compare to this oracle.
- More **review questions + deterministic checks**; promote `DUP-001`/`CONS-001`
  from `llm_judge` to deterministic now that fusion is structured.

## Done
- ✅ contracts · prompts (DRAFT) · review framework · rubric guard
- ✅ **shared deterministic perception** (`tools.py`) — VLM tool slots + engine share it
- ✅ **A1 real imaging** — BGR + Kirchhoff/LSM migration (`gpr_data_processing.migration`)
  → a true subsurface section, so A3 detects on focused apex energy (A3 unchanged).
  On TU1208 the *same* A3 goes 0 → 4 GT-depth hits; loop recall 0.62 → 0.85.
- ✅ **keyless A1→A2‖A3→A4 loop** (`deterministic.py`) with cross-domain de-dup +
  bounded re-plan → real `InterpretationCase`, scored vs GPR-Bench + reviewed

# GPR Agent — target design (4 agents)

The intended architecture (supersedes `gpr_interpretation`'s linear
perception→hypothesis→grounding→reporter chain; reuses its building blocks). It
is **domain-specialized + orchestrated**, which is what enforces *no
double-counting* and *evidence independence*.

## First principles

**1. No calibration (yet) — say "relative", never "calibrated".** This is the
foundational principle; it gates every claim the agents and the docs make.

- **No parameter is fit to ground truth.** Detector thresholds, tolerances,
  velocity, and re-plan budgets are *physically-motivated defaults*. GT is used to
  **evaluate**, never to **tune**. Any GT-tuned number requires a held-out
  train/test split — and the bundled datasets are **one line each**, so tuning on
  them and reporting on them is **leakage**. Don't.
- **No uncertainty calibration.** `confidence` is a **relative rank**, not a
  probability; velocity is **nominal** (`velocity_calibrated=False`) until a
  physics autofocus or a coverage/ECE check (`gpr_bench.scoring_uq` on held-out
  data) *earns* the upgrade. Only that turns "relative" into "calibrated".
- **Therefore every reported metric is a dev sanity bar**, not a tuned or
  calibrated result, and the prose must label it as such.

(The other rubric dimensions — consistency, physical plausibility, no circular
reasoning, no double-count, no overclaim, evidence-based — sit *under* this one:
they constrain the agents' reasoning, but no-calibration honesty constrains what
may be *claimed*.)

```
            ┌──────────────────────────────────────────────────────────┐
            │  A4  LEADER / ORCHESTRATOR / REVIEWER / INTERPRETER        │
            │  plan ▸ dispatch ▸ fuse ▸ interpret ▸ gate (human review)  │
            │  RUBRIC: consistent · physically plausible · calibrated    │
            │  uncertainty · no circular reasoning · no double-count ·   │
            │  no overclaim · evidence-based                             │
            └──────┬──────────────────┬──────────────────┬──────────────┘
          plan/params│         dispatch│          dispatch│
                     ▼                 ▼                  ▼
          ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
          │ A1 IMAGING     │  │ A2 RADARGRAM   │  │ A3 SUBSURFACE  │
          │ preprocessing  │  │ analyzer /     │  │ analyzer /     │
          │ + subsurface   │─▶│ detector       │  │ detector       │
          │ imaging        │  │ (B-scan, VLM   │  │ (model image,  │
          │ (tool-augmented)│ │  + tools)      │  │  VLM + tools)  │
          └──────┬─────────┘  └──────┬─────────┘  └──────┬─────────┘
                 │ B-scan + model    │ radargram          │ subsurface
                 │ (+ params)        │ evidence           │ evidence
                 └───────────────────┴────────────────────┘ ─▶ up to A4
```

## The four agents

| Agent | Role | Domain | Kind | Tools (from `gpr_data_processing`) | Output contract |
|---|---|---|---|---|---|
| **A1 Imaging** | produce the two image domains | raw → B-scan + model | tool-augmented (agentic param choice) | `preprocessing` (bandpass/gain/declutter/time-zero), `migration` (Kirchhoff/LSM/PnP) | `PreprocessedBScan` + `SubsurfaceImage` + **imaging param provenance** |
| **A2 Radargram analyzer/detector** | detect/analyze on the B-scan | radargram | **tool-augmented VLM** | `attributes` (dip, FK, radon, envelope), `detection` (blob/peak, hyperbola fit, semblance) | `RadargramEvidence` (features in B-scan coords + confidence + provenance) |
| **A3 Subsurface analyzer/detector** | detect/analyze on the imaged model | subsurface model | **tool-augmented VLM** | `detection` (blob NMS, depth bands, line features), `materials` (plausibility on the focused image) | `SubsurfaceEvidence` (targets/boundaries in depth coords + confidence + provenance) |
| **A4 Leader** | plan, orchestrate A1–A3, fuse, interpret, gate | cross-domain | **tool-augmented VLM** + the rubric | `uncertainty` (stats), `materials` + `agent_validators` (plausibility/consistency) | `InterpretationCase` (final evidence-based interpretation + audited rubric) |

A1/A2/A3 communicate **only** through the typed contracts → replayable, auditable,
and their evidence stays **independent** (A4's conclusions never re-enter A2/A3).

## A4's rubric (the heart of the intent — this is *validation*, not fitting)
| Guarantee | How A4 enforces it |
|---|---|
| **Consistent** | reconcile A2 (B-scan) ↔ A3 (model) for the *same* physical feature; surface disagreement, never hide it |
| **Physically plausible** | εr / velocity / depth / material / diameter within bounds (`materials` + `agent_validators` + `gpr_data_processing.validation`) |
| **Calibrated uncertainty** | propagate confidence/uncertainty (`gpr_data_processing.uncertainty`), but treat it as **uncalibrated until verified** — UQ is *not* statistical correctness on its own; calibration (coverage / reliability on GT, via GPR-Bench `scoring_uq`) must pass before the numbers are trusted. No "best-of-many-tries" without correction. |
| **No circular reasoning** | site context = *uncertain prior*, never ground truth; A4's interpretation is not fed back as independent evidence |
| **No double-count** | **cross-domain de-dup**: a hyperbola apex (B-scan) and a focused blob (model) at the same (x, depth) = **one** target — the reason A2/A3 are split |
| **No overclaim** | tentative interpretation only; never "confirmed utility / clearance decision"; confidence-graded |
| **Evidence-based** | every claim traces to tool evidence + the provenance ledger |

## How it reuses what exists (nothing built from scratch)
- **`gpr_interpretation` (SimV3):** keep its LLM backends, pydantic contracts, and the 40-tool catalog — **re-topologized**: split perception_tools across A1 (imaging), A2 (radargram), A3 (subsurface); fold hypothesis+grounding+reporter+planner into **A4**.
- **`agent_validators` (GPR-Tools sibling):** A4's plausibility/consistency/provenance contracts.
- **`gpr_data_processing` (GPR-Tools):** the tools for all four (preprocessing/migration → A1; attributes/detection → A2/A3; uncertainty/materials/validation → A4).
- **`gpr_viz`:** render the two domains + overlays (for the VLMs and the UI).
- **`temp_src` prototype:** the **palette-flip ambiguity probe** → A2's diagnose step; the **two-domain session** → A1's output shape; the **no-LLM deterministic sandbox** → a test harness for the topology (and the UI's *human/manual* mode).

## Alignment to the ui-mock (the latest design)
- **Dual panels** (radargram + subsurface) = A2's and A3's working surfaces; A1 fills both.
- **Stage rail** `diagnose → plan → implement → validate` = A4 plans, A1 implements imaging, A2/A3 implement detection, A4 validates (the rubric).
- **Human ⇄ agent toggle** = the deterministic/manual path (human drives the tools) vs A4-orchestrated; human can override any agent.
- **Provenance/step ledger** = the typed-contract handoffs + param provenance.

## Harness engineering (A4 ⇄ A1/A2/A3) — the interesting part
The topology is easy; the harness is where the rubric is *structurally* enforced.

**Pattern:** orchestrator-workers over an **append-only, provenance-stamped
evidence store** (a blackboard). A4 = orchestrator; A1/A2/A3 = workers. The store
is the single source of truth + the replay log + the UI's ledger.

**Control flow (bounded — "agent at decision boundaries only"):**
1. A4 *plans* (imaging strategy + which analyses) → dispatches **A1**.
2. A1 emits B-scan + subsurface image + **velocity model** + param provenance → store.
3. A4 fans out **A2 ∥ A3 in parallel** → each runs its tool-use loop → posts Evidence.
4. A4 fans in → **fuse** (common-frame association) → **rigor gates** → interpret.
5. Optional **bounded re-plan** (≤ N rounds): A4 may re-image (new velocity) or ask
   a worker to *re-examine region R* — a **Query, never a conclusion**.
6. **Human-review gate** (HITL) before any acceptance.

**Directional contracts (this is how circular reasoning is killed):**
- A4 → worker = `Task`/`Query` (*"analyze this image" / "re-look at R"*) — never a finding.
- worker → A4 = `Evidence` (typed, provenance-stamped, confidence) — **up only**.
- **A2 ⟂ A3**: workers never read each other or A4's interpretation. Fusion happens
  *only* at A4. Independence is enforced by the dispatch API's types, not by convention.

**Per-worker loop (A2/A3/A4 are VLMs):** `call_with_tools` — the VLM sees the
*rendered* domain image + a tool catalog, emits `{tool, args}`, the harness runs it
against a Python **session** (the arrays the VLM never sees → small message history),
returns a `ToolResult` (numbers + a render), repeats to a **budget**, then yields
typed `Evidence` + provenance. (Reuse `gpr_interpretation`'s `ToolUsePerceptionAgent`
+ `LLMBackend.call_with_tools`.)

**Cross-domain fusion (no double-count):** A2 reports in (trace, twtt); A3 in
(x, depth). A4 projects A2 → depth via A1's velocity model, then **associates** A2/A3
in the common (x, depth) frame (distance/IoU tolerance, greedy or Hungarian). A match
= **one** target (cross-confirmed ⇒ higher confidence); an unmatched item = single-
domain (flagged, lower confidence). De-dup is the harness's job, not the workers'.

**Rigor gates = deterministic validators A4 runs over the store** (it *consults*, it
can't silently override — every decision is logged): plausibility (`materials` /
`validation` / `agent_validators`), **calibration** (`uncertainty` numbers checked
for coverage/reliability via GPR-Bench `scoring_uq` — *not assumed correct*), de-dup
(association), consistency (reconcile A2↔A3 disagreement), overclaim guard
(confidence caps + "tentative" phrasing).

**Replayability / testing:** append-only store + recorded tool calls/results ⇒
deterministic replay; a `StubBackend` (canned VLM responses) tests the wiring without
an API; the `temp_src` no-LLM sandbox is the deterministic harness skeleton + the UI's
*human/manual* mode.

**Hard bits worth the effort:** (1) typed dispatch API that makes circular reasoning
structurally impossible; (2) the time↔depth common-frame association (de-dup +
cross-confirm); (3) bounded re-planning without loops; (4) small-message session vs
full provenance; (5) parallel A2∥A3 with partial-failure graceful degradation;
(6) VLM budget/cost control; (7) clean HITL override points at every gate.

## A4 reasoning — hypothesis-driven, evidence-based (the reviewer's mind)
A4 does **not** just bottom-up fuse A2/A3; it runs a bounded **hypothetico-deductive
loop** over a live set of *competing* hypotheses (reuse `SubsurfaceHypothesis`).
The rubric then *falls out* of the reasoning rather than being bolted on.

1. **Abduce** — from the fused A2/A3 evidence, propose candidate explanations
   (pipe / layer boundary / void / rebar / ringing-artifact …), each with a prior.
2. **Predict** — derive what each hypothesis *should* produce **in each domain**
   (e.g. a metal pipe ⇒ a strong phase-inverted hyperbola in the B-scan **and** a
   focused bright point in the migrated image at the matching depth).
3. **Test** — check predictions against the *independent* A2/A3 evidence; if
   insufficient, dispatch a **falsifiable Query** ("re-examine region R") —
   **never** "there is a pipe at R" (keeps the workers' evidence independent).
4. **Update** — move belief with likelihoods + uncertainty
   (`gpr_data_processing.uncertainty`); keep competing hypotheses alive (no
   premature collapse). The uncertainty is **uncalibrated until verified** — a
   confidence is a *relative* ranking signal, not a calibrated probability, until
   the calibration gate passes (coverage on GT).
5. **Adjudicate** — accept / reject / mark ambiguous via the rigor gates.

How this *is* the rubric:
- **Evidence-based / no overclaim** — belief = Σ traced evidence; hypotheses stay
  confidence-graded (candidate → supported → tentative), never "confirmed utility".
- **No circular reasoning** — predictions are tested against workers that never saw
  the hypothesis, and A4 actively seeks **refuting** evidence (falsification, not
  confirmation bias).
- **No double-count** — each evidence item supports a hypothesis once; a cross-domain
  match is *one* independent confirmation (via the association), not two.
- **Consistent / physically plausible** — a surviving hypothesis must explain **both**
  domains coherently and stay within physical bounds (`materials` / `validation`).

This also drives the bounded re-plan: A4 re-dispatches A1/A2/A3 *because a hypothesis
needs a specific, falsifiable test* — that is the "agent at decision boundaries" call.

## Validation loop (wired)
`run_agent_validation.py` wires the benchmark to the agent:
**GPR-Bench dataset (radargram + GT) → `perceive(...)` → `gpr_bench.eval`**. The
agent's A2/A3 detectors are the `perceive` plug point. Two methods run today:
a GT-independent **baseline** (mean-envelope depth-peaks) and a first
**deterministic A2** (max-across-traces apex-depth detection). On TU1208 the
deterministic A2 already beats the baseline:

| method | precision | recall | AP | depth-MAE |
|---|---|---|---|---|
| baseline (envelope stack) | 0.30 | 0.46 | 0.33 | 0.050 m |
| **A2 radargram (apex depths)** | **0.40** | **0.62** | **0.39** | 0.062 m |

The VLM/tool-use A2 (and A3/A4) plug in at the same `Detection` contract and must
beat this in turn.

## Pydantic-AI draft
`gpr_agent/` expresses this in [Pydantic-AI](https://ai.pydantic.dev):
`contracts.py` (typed handoffs), `agents.py` (the 4 agents + `orchestrate()` +
a `TestModel` no-key demo), and `review.py` (an **extensible review-question bank**
+ deterministic checks operationalizing the rubric). Prompts are `[DESIGN ME]`
slots — the careful work. See `gpr_agent/README.md`.

## Status
Design + the validation loop + the Pydantic-AI scaffold. The 4 agents' prompts/
tools are not implemented; nothing promoted.

**Home: incubated in the workbench** — it grows in `experiments/agent/` and
graduates to `apps/agent` (product face) when it matures. **No separate
`GPR-Agent` repo for now** (the workbench is the incubator); extract to its own
repo only if it outgrows the incubator.

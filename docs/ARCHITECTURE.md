# GPR-Agent — Architecture

The interpretation **application** of the GPR platform: raw radargram → reviewed,
diagnosed `InterpretationCase`. This doc is the architectural overview; the deep design
rationale (why these boundaries, the contracts, the comparison of design drafts) lives in
[`AGENTS_DESIGN.md`](AGENTS_DESIGN.md).

## First principle — no calibration
Nothing in this system is fit to ground truth.
- **Confidences are relative ranks, not probabilities.** "0.8" means "ranked above 0.6
  here", not "80% likely".
- **Velocity is nominal**; depths are nominal-velocity conversions, reported as such.
- **GT evaluates, it never tunes.** The bundled dev lines are one example each — selecting
  or tuning a tool because it helps that single line is leakage (n=1 selection bias). New
  capabilities are added as *separate* orchestrators; the deterministic oracle is frozen.
- Results are a **dev sanity bar**, never "calibrated". The only GT-free self-tuning
  allowed is autofocus on a focus metric (which uses no labels).

## The four agents
```
   A1  imaging      BGR + Kirchhoff migration → the images all downstream agents read
     │              (migration is A1's job; A2/A3 only detect + analyse)
 ┌───┴────┐
 A2        A3       run in PARALLEL, each a single domain:
 radargram subsurface
 1-D apex   2-D blob NMS,     directional contracts (one-way):
 profile,   spatial layout      • DOWN: Task / Query (A4 → A2,A3)
 polarity,                       • UP:   DomainEvidence (A2,A3 → A4)
 amplitude,                    so the domains never couple to each other.
 ringing
 └───┬────┘
   A4  leader       HYBRID:
                     (1) deterministic geometric FUSE — gated greedy bipartite
                         (x, depth) association → de-dup → falsification → bounded
                         re-plan. NOT a prompt, NOT a matched filter: pure geometry.
                     (2) a VLM narrative over the fused case.
                     (3) a CONTROL loop that diagnoses upstream problems and acts.
```

### Two kinds of diagnosis
- **Self / control** (`control.py`) — A4 inspects the run: a GT-free focus metric flags
  defocus → it autofocuses (re-migrates at the best-focus velocity) and re-runs A2/A3/A4.
  Designed to extend to clutter → re-declutter, low-SNR → re-gain, tool-mismatch → reroute.
- **Target** (`materials.py`) — per hypothesis, the radargram polarity (+ an optional,
  *do-no-harm-gated* amplitude/conductor cue) and depth map to a **candidate material/kind
  set** read from the canonical KB. Honest by construction: polarity *narrows* (metal and
  air-void both read negative); it does not uniquely identify. Amplitude is a low-reliability
  prior (gain/depth/attenuation confound it), so only positive, high-confidence evidence
  narrows — absence never excludes.

## The keyless deterministic oracle
`deterministic.py` runs the **entire topology with no LLM and no API key** — the shared
`tools.py` detectors feed the geometric A4 fuse, emitting a real `InterpretationCase`. It
is the reference the VLM agents are measured against, and it is **FROZEN**: `FROZEN_PARAMS`
+ `test_oracle_frozen.py` pin its signature on a fixed synthetic scene, so refactors can't
silently move it. New behaviour goes in *new* orchestrators, never by editing the oracle.

## NL-instructed simulation (the physics-truth flywheel)
`sim_scenes.py` turns a parametric / natural-language scene description into FDTD truth:
```
Scene (add_pipe / add_void / add_layer)
   → build123d CAD (OpenCASCADE solids)
   → OCP point-in-solid voxelization
   → gprMax geometry HDF5 (/data int volume + dx_dy_dz) + materials.txt
   → #geometry_objects_read → gprMax GPU B-scan  (per-trace single-model runs)
```
EM properties come from the **canonical KB** (one source for "PVC ε_r" / "what is a void").
GPU is driven through GPR-Sim's runtime (CUDA 12.x). This gives labeled, physics-true
radargrams to develop and sanity-check the agents against — *without* fitting to them.

## Where it sits in the platform
GPR-Agent is the **L3 application** — the top of the stack, depending downward only:
```
  L0 leaves:  GPR-Viz            GPR-KnowledgeBase (reference: materials + taxonomy)
  L1 libs:    GPR-Tools   GPR-Sim   GPR-CV
  L2 eval:    GPR-Bench
  L3 app:     GPR-Agent  ──►  GPR-Tools + GPR-KB + GPR-Bench (+ GPR-Sim for FDTD)
```
- **GPR-Tools** — perception engine (BGR, migration, envelope, `estimate_apex_polarity`,
  `estimate_apex_ringing`, autofocus).
- **GPR-KB** — the reference library, consumed via data contracts (`gpr_kb.reference`).
- **GPR-Bench** — scoring the case vs GT (evaluation only; never feeds back into tuning).
- **GPR-Sim** *(optional)* — the gprMax runtime for `sim_scenes`.

A boundary test forbids importing the workbench. Dependencies point **down only**.

## Dev discipline
- **Leakage-free dev loop** — refine prompts/params against a process rubric +
  oracle-agreement on dev/synthetic scenes, never against the held-out TU1208 line.
- **Frozen oracle** — the deterministic reference never drifts.
- **Honest abstention** — tools report `AMBIGUOUS` / `has_baseline=False` / "narrows, not
  unique" rather than guessing; the VLM narrative is constrained to the geometric fuse.

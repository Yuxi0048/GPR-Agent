# GPR-Agent

**Agentic GPR interpretation** — a 4-agent workflow that turns a raw radargram into a
reviewed, diagnosed `InterpretationCase` (where the buried objects are, and *what* they
plausibly are), built on the [GPR platform](https://github.com/Yuxi0048/GPR-KB) libraries.
It ships with a **keyless deterministic oracle** (the whole topology, no LLM, no API key)
and an **NL-instructed simulation path** (a natural-language / parametric scene → CAD →
FDTD B-scan) that supplies physics-truth scenes to develop and sanity-check against.

> **First principle — no calibration.** Nothing is fit to ground truth. Confidences are
> *relative ranks*, not probabilities; velocity is nominal. Results are a **dev sanity
> bar**, never "calibrated". Ground truth *evaluates*, it does not *tune* — bundled dev
> lines are one example each, so tuning-then-reporting on them would be leakage.

## The four agents
```
            A1  imaging        (BGR + Kirchhoff migration; produces the images)
              │
       ┌──────┴───────┐
   A2 radargram    A3 subsurface   (run in parallel; directional contracts:
   (1-D apex,       (2-D blob NMS,   Task/Query down, DomainEvidence up)
    polarity,        spatial layout)
    amplitude,
    ringing)
       └──────┬───────┘
            A4  leader        (HYBRID: deterministic geometric fuse — gated bipartite
                               de-dup + falsification + bounded re-plan — PLUS a VLM
                               narrative; also a control loop that DIAGNOSES upstream
                               problems, e.g. defocus → autofocus → re-run)
```

Two senses of *diagnosis*: **self/control** (A4 detects defocus/low-SNR/clutter and
re-routes or re-runs upstream) and **target** (per hypothesis: polarity + amplitude +
depth → candidate material/kind, read from the canonical KB — honest, *narrows* not IDs).

## Layout
```
src/gpr_agent/
  contracts.py      typed inter-agent handoffs (pydantic)
  session.py        shared Session (arrays / renders)
  prompts/          the 4 system prompts as reviewable markdown
  tools.py          deterministic perception (numpy/scipy + GPR-Tools)
  deterministic.py  the KEYLESS A1→(A2‖A3)→A4 oracle  (FROZEN + pin-tested)
  control.py        A4 diagnostic control loop (autofocus re-run)
  materials.py      polarity → candidate material/kind  (reads gpr_kb.reference)
  agents.py         the 4 agents in Pydantic-AI  (optional: pip install .[vlm])
  review.py / review_llm.py   review-question bank + deterministic checks + LLM judge
  render.py         Session image → PNG
  dev_scenes.py     synthetic dev scenes (GT-free sanity)
  sim_scenes.py     NL/parametric scene → build123d CAD → gprMax FDTD  (optional: .[sim])
  test_*.py         the suite (deterministic oracle pinned; materials; review; control)
scripts/            runnable loops (deterministic / vlm / polarity-material / refine-dev)
docs/               AGENTS_DESIGN.md (the design), COMPARISON.md, ARCHITECTURE.md
```

## Where it sits
GPR-Agent is the **L3 application** at the top of the stack. It depends *downward* only:

| consumes | for |
|---|---|
| **GPR-Tools** (`gpr_data_processing`) | the perception engine — BGR, migration, envelope, polarity, ringing, autofocus |
| **GPR-KB** (`gpr_kb`) | the canonical reference library — materials/EM props + buried-object taxonomy |
| **GPR-Bench** (`gpr_bench`) | scoring the case against ground truth (evaluation only) |
| **GPR-Sim** (`subsurface_platform`) *(optional)* | the gprMax runtime for the FDTD simulation path |

A boundary test (`test_boundary.py`) enforces it never imports the workbench. See the
cross-repo big picture in **[GPR-KnowledgeBase/ARCHITECTURE.md](https://github.com/Yuxi0048/GPR-KB)**.

## Install & run
```bash
pip install -e .            # core (numpy/scipy/pydantic) — the keyless oracle runs
pip install -e .[vlm]       # + pydantic-ai for the A1–A4 VLM agents
pip install -e .[sim]       # + build123d for the FDTD scene path (gprMax via GPR-Sim)

# the platform libs are resolved via editable installs of the sibling repos, or
#   PYTHONPATH="…/GPR-Tools/src;…/GPR-KnowledgeBase;…/GPR-Bench"
pytest src/gpr_agent -q                       # the suite (no key needed)
python scripts/run_deterministic_loop.py tu1208   # the keyless end-to-end oracle
```

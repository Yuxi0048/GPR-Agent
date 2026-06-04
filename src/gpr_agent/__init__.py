"""GPR-Agent — the 4-agent GPR interpretation system (A1 imaging → A2 radargram ‖
A3 subsurface → A4 leader), built on the GPR platform libraries.

- ``contracts``     -- typed inter-agent handoffs (pydantic; runs anywhere).
- ``session``       -- the shared Session (arrays/renders); no heavy deps.
- ``prompts``       -- the 4 system prompts as reviewable markdown (no pydantic-ai).
- ``tools``         -- deterministic perception tool bodies (numpy/scipy + gpr_data_processing).
- ``deterministic`` -- the KEYLESS A1->(A2||A3)->A4 loop (no LLM) -> InterpretationCase.
- ``control``       -- A4 diagnostic control loop (defocus -> autofocus -> re-run).
- ``materials``     -- polarity -> candidate material/kind (reads the canonical KB).
- ``agents``        -- the 4 agents in Pydantic-AI format (needs ``pip install pydantic-ai``).
- ``review``        -- extensible review-question bank + deterministic checks (pydantic).
- ``sim_scenes``    -- NL/parametric scene -> build123d CAD -> gprMax FDTD B-scan (optional).

FIRST PRINCIPLE — no calibration: nothing is fit to ground truth; confidences are
relative ranks, not probabilities; velocity is nominal. Results are a *dev sanity bar*,
never "calibrated". GT evaluates, it does not tune.

``agents`` is intentionally NOT imported here (pydantic-ai is optional) -- import it
directly when the library is installed. ``tools``/``deterministic`` import numpy lazily-ish.
"""
__version__ = "0.1.0"

from . import contracts, control, deterministic, dev_scenes, materials, prompts, render, review, session, tools  # noqa: F401,E402

"""System prompts for the 4 agents, kept as reviewable markdown (DRAFT v0.1).

Loadable WITHOUT pydantic-ai so prompts can be inspected, diffed, and unit-tested
standalone (e.g. assert the discipline keywords are present). agents.py imports
these as the agents' `system_prompt`. Grown from the legacy linear pipeline in
../../from_simulationv3/gpr_interpretation/llm/prompts/ (perception -> A1+detectors,
hypothesis+grounding -> A4), re-shaped into the directional A1->(A2||A3)->A4 design.
"""
from pathlib import Path

_DIR = Path(__file__).parent

NAMES = {
    "a1": "a1_imaging",
    "a2": "a2_radargram",
    "a3": "a3_subsurface",
    "a4": "a4_leader",
}


def load(agent: str) -> str:
    """Return the system-prompt text for an agent key ('a1'..'a4') or a file stem."""
    stem = NAMES.get(agent, agent)
    return (_DIR / f"{stem}.md").read_text(encoding="utf-8")


A1 = load("a1")
A2 = load("a2")
A3 = load("a3")
A4 = load("a4")

__all__ = ["load", "NAMES", "A1", "A2", "A3", "A4"]

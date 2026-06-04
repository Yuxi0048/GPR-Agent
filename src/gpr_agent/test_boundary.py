"""Fast-fail boundary test: GPR-Agent must not import the workbench.

GPR-Agent is the TOP application layer (L3). It may depend DOWNWARD on the platform
libraries — GPR-Tools (``gpr_data_processing``), GPR-KB (``gpr_kb``), GPR-Bench
(``gpr_bench``), GPR-Sim (``subsurface_platform``), GPR-Viz (``gpr_viz``) — but it must
NOT import the workbench sandbox/app, which sits beside the stack and pins the libs.
Only real import statements are flagged (docstring mentions are fine).
"""
from __future__ import annotations

import pathlib
import re

import gpr_agent

FORBIDDEN = ("gpr_workbench", "data_helper")


def _imports(text: str, name: str) -> bool:
    return bool(re.search(rf"(?m)^\s*(?:from|import)\s+{re.escape(name)}\b", text))


def test_package_has_version():
    assert gpr_agent.__version__


def test_no_upward_imports():
    pkg = pathlib.Path(gpr_agent.__file__).parent
    offenders = []
    for py in pkg.rglob("*.py"):
        if py.name.startswith("test_"):
            continue
        text = py.read_text(encoding="utf-8")
        offenders += [f"{py.name}: {b}" for b in FORBIDDEN if _imports(text, b)]
    assert not offenders, f"upward imports found: {offenders}"

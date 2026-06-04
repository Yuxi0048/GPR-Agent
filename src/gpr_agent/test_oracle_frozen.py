"""Pin test: the deterministic oracle is FROZEN -- its behaviour must not drift.

The oracle is a fixed reference (see deterministic.py banner + AGENTS_DESIGN first
principle). This test runs it on a FIXED synthetic input and asserts a stable output
signature + run-to-run determinism. A casual parameter change (the kind that would be
n=1 selection bias) flips these numbers and fails the test -- forcing a deliberate,
documented decision instead of silent tuning. Uses migrate=False (no pylops, fast):
the pin is on the detection + fusion LOGIC, which is where tuning pressure lives.
"""
from __future__ import annotations

import pytest

pytest.importorskip("gpr_data_processing", reason="needs the envelope engine")

from .contracts import HypothesisStatus              # noqa: E402
from .deterministic import FROZEN_PARAMS, orchestrate_deterministic  # noqa: E402
from .session import Session                          # noqa: E402
from .test_deterministic import _synthetic           # noqa: E402

# Recorded signature of the frozen oracle on _synthetic(seed=0), migrate=False.
# Changing these means the oracle changed -- update ONLY with a documented reason.
_EXPECTED = {"n_hyp": 40, "n_confirmed": 20, "n_supported": 20, "n_refuted": 1, "rounds": 1}


def _run():
    s = Session(arrays={"raw": _synthetic(seed=0)})
    return orchestrate_deterministic(s, "raw", dt_ns=0.1, dx_m=0.02, max_rounds=2, migrate=False)


def _signature(res) -> dict:
    h = res.case.hypotheses
    return {
        "n_hyp": len(h),
        "n_confirmed": sum(x.cross_domain_confirmed for x in h),
        "n_supported": sum(x.status == HypothesisStatus.supported for x in h),
        "n_refuted": sum(x.status == HypothesisStatus.refuted for x in h),
        "rounds": res.rounds,
    }


def test_oracle_signature_is_frozen():
    assert _signature(_run()) == _EXPECTED, "the frozen oracle changed -- tuning is a non-goal; document why"


def test_oracle_is_deterministic():
    a = [(round(x.x_m, 4), round(x.depth_m, 4), x.confidence, x.status.value) for x in _run().case.hypotheses]
    b = [(round(x.x_m, 4), round(x.depth_m, 4), x.confidence, x.status.value) for x in _run().case.hypotheses]
    assert a == b, "oracle output differs across identical runs -- it must be deterministic"


def test_frozen_params_are_documented():
    # the canonical param table exists and records the nominal (uncalibrated) velocity
    assert FROZEN_PARAMS["velocity_m_per_ns"] == 0.1
    assert {"A3_thresh_pct", "fuse_dz_tol_m", "max_rounds"} <= set(FROZEN_PARAMS)

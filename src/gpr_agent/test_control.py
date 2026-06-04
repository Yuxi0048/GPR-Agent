"""A4 diagnostic control loop: a wrong nominal velocity is DIAGNOSED (defocused
migration) and AUTOFOCUS-corrected (GT-free), then the pipeline re-runs. Needs the
gpr_data_processing migration engine; runs a few Kirchhoff migrations (~seconds)."""
from __future__ import annotations

import pytest

pytest.importorskip("gpr_data_processing", reason="needs the migration + autofocus engine")
pytest.importorskip("pylops", reason="migration needs the [migration] extra")

from .control import orchestrate_with_control   # noqa: E402
from .dev_scenes import synth_scene             # noqa: E402
from .session import Session                    # noqa: E402


def _run(nominal_v: float):
    img, dt, dx, _ = synth_scene(seed=0, n_s=160, n_t=120)   # smaller for a fast migration sweep
    return orchestrate_with_control(Session(arrays={"raw": img}), "raw", dt_ns=dt, dx_m=dx, velocity=nominal_v)


def test_wrong_nominal_velocity_is_diagnosed_and_autofocus_corrected():
    res, actions = _run(0.15)                    # planted moveout velocity is 0.1 m/ns
    assert actions, "a wrong nominal velocity should be diagnosed + acted on"
    a = actions[0]
    assert a.symptom == "defocused migrated section"
    assert "autofocus" in a.action
    assert a.metric_after > a.metric_before      # focus genuinely improved


def test_autofocus_recovers_the_true_velocity_and_marks_calibrated():
    res, _ = _run(0.15)
    assert res.imaging.velocity_calibrated is True            # earned by focus, not GT
    assert 0.075 <= res.imaging.velocity_m_per_ns <= 0.125    # recovered ~true 0.1 from focus alone


def test_control_loop_emits_a_valid_interpretation_case():
    res, _ = _run(0.15)
    assert res.case.hypotheses and res.case.is_tentative is True
    assert "autofocus-CALIBRATED" in res.case.overall_note

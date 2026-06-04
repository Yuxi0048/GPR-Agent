"""Polarity -> candidate-material lookup (reads the canonical KB). The FDTD-truth
validation lives in ../run_polarity_material.py; this guards the mapping + discipline.
Needs gpr_kb (the KnowledgeBase) on the path."""
from __future__ import annotations

import pytest

pytest.importorskip("gpr_kb", reason="needs GPR-KnowledgeBase (gpr_kb) on the path")

from . import materials                          # noqa: E402
from .contracts import Hypothesis                # noqa: E402


def test_sign_mapping_handles_words_and_ints():
    assert materials.polarity_sign("NEGATIVE") == -1
    assert materials.polarity_sign("POSITIVE") == 1
    assert materials.polarity_sign("AMBIGUOUS") == 0 and materials.polarity_sign(None) == 0
    assert materials.polarity_sign(-1) == -1 and materials.polarity_sign("reversed") == -1


def test_negative_polarity_set_includes_conductor_and_void():
    # NEGATIVE is conductor-dominated up top; the void is a NEGATIVE candidate too,
    # so check the full set (top=20) -- exactly what run_polarity_material validated.
    ids = [sid for sid, _ in materials.candidates("NEGATIVE", 10.0, top=20)]
    assert "steel_pipe_empty" in ids and "air_void" in ids   # both reversed (the validated cases)


def test_positive_polarity_set_includes_water():
    ids = [sid for sid, _ in materials.candidates("POSITIVE", 10.0)]
    assert "water_pocket" in ids


def test_best_material_for_negative_is_a_conductor_first():
    assert materials.best_material("NEGATIVE", 10.0) == "steel_pipe_empty"


def test_annotate_sets_material_and_notes_nonuniqueness():
    h = Hypothesis(id="H1", statement="point reflector")
    materials.annotate_hypothesis(h, "NEGATIVE", 10.0)
    assert h.material is not None
    assert "candidate material" in h.statement and "not unique" in h.statement


def test_ambiguous_polarity_does_not_annotate():
    h = Hypothesis(id="H1", statement="point reflector")
    materials.annotate_hypothesis(h, "AMBIGUOUS", 10.0)
    assert h.material is None


def test_diagnose_negative_apex_narrows_to_pipe_or_void_not_unique():
    h = Hypothesis(id="H1", statement="point reflector", depth_m=0.8)
    materials.diagnose_hypothesis(h, "NEGATIVE", 10.0)   # conductors first -> steel pipe is 'best'
    assert h.kind == "pipe" and h.material == "steel_pipe_empty"
    # honest: NEGATIVE narrows to {pipe, void}, not a unique ID
    assert "candidate kinds" in h.statement and "void" in h.statement
    assert "cannot uniquely ID" in h.statement


def test_diagnose_positive_apex_is_water_bearing():
    # POSITIVE -> high-eps target -> water-bearing (water-filled pipe / water pocket)
    h = Hypothesis(id="H1", statement="point reflector", depth_m=1.0)
    materials.diagnose_hypothesis(h, "POSITIVE", 10.0)
    ids = [sid for sid, _ in materials.candidates("POSITIVE", 10.0, top=10)]
    assert "water_pocket" in ids                     # water pocket is a POSITIVE candidate
    assert "water" in h.material                      # best is water-bearing (e.g. *_pipe_water)
    assert h.kind in ("pipe", "void")


def test_diagnose_flags_depth_outside_typical_range():
    h = Hypothesis(id="H1", statement="point reflector", depth_m=0.15)   # shallow (FDTD/dev)
    materials.diagnose_hypothesis(h, "NEGATIVE", 10.0)
    assert "outside typical" in h.statement   # steel typical 0.3-5.0 m -> 0.15 flagged, not excluded


def test_conductor_amplitude_cue_pins_metal_pipe():
    # NEGATIVE + strong + CONDUCTOR -> metal pipe, void excluded (the amplitude cue resolves it)
    h = Hypothesis(id="H1", statement="apex", depth_m=0.8)
    materials.diagnose_hypothesis(h, "NEGATIVE", 10.0, conductor=True, amplitude="strong")
    assert h.kind == "pipe" and h.material == "steel_pipe_empty"
    kinds_clause = h.statement.split("candidate kinds {")[1].split("}")[0]
    assert "void" not in kinds_clause                     # conductor-narrowed -> all pipe


def test_weak_or_absent_amplitude_cue_does_no_harm():
    # amplitude is UNRELIABLE: a 'weak' reading / absent conductor (no ringing) must NOT
    # exclude metal -- a deep/attenuated metal also reads weak. So the diagnosis falls back
    # to the polarity-based best, undamaged.
    h = Hypothesis(id="H1", statement="apex", depth_m=0.8)
    materials.diagnose_hypothesis(h, "NEGATIVE", 10.0, conductor=False, amplitude="weak")
    assert h.material == "steel_pipe_empty"               # do-no-harm: metal not excluded


def test_kind_is_canonical_from_kb_not_a_heuristic():
    # the workbench reads gpr_kb's kind; e.g. clay_lens -> 'lens' (a keyword guess would miss it)
    assert materials._kind_of("clay_lens") == "lens"
    assert materials._kind_of("bedding_contact") == "layer"


def test_diagnose_case_uses_detection_polarity():
    from .contracts import Detection, DomainEvidence, InterpretationCase, Provenance
    det = Detection(id="r000", domain="radargram", x_m=2.0, depth_m=0.8, polarity="NEGATIVE",
                    supporting_tools=["envelope"])
    ev = [DomainEvidence(domain="radargram", detections=[det], provenance=Provenance(agent="A2"))]
    case = InterpretationCase(hypotheses=[Hypothesis(id="H1", statement="pt", depth_m=0.8,
                              supporting_evidence=["r000"])], provenance=Provenance(agent="A4"))
    materials.diagnose_case(case, ev, 10.0)
    assert case.hypotheses[0].kind == "pipe"

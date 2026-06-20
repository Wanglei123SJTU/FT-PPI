from __future__ import annotations

from src.experiments.upworthy_lowdim_feature_screening import build_lowdim_specs


def test_build_lowdim_specs_enumerates_target_only_pairs_and_final_controls():
    specs = build_lowdim_specs(
        targets=["delta_A", "delta_B"],
        control_pool=["delta_A", "delta_B", "delta_C"],
        final_controls=["delta_A", "delta_C"],
        dimensions=[1, 2],
    )
    by_id = {spec.spec_id: spec for spec in specs}
    assert "A" in by_id
    assert "A__B" in by_id
    assert "A__C" in by_id
    assert "B__A" in by_id
    assert "B__C" in by_id
    assert "B__A__C" in by_id
    assert by_id["A"].feature_columns == ("delta_A",)
    assert by_id["B__A__C"].family == "final_controls"

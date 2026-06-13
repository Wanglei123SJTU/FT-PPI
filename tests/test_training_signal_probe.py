from __future__ import annotations

from src.experiments.run_training_signal_probe import build_probe_configs


def test_build_probe_configs_merges_base_and_variants():
    config = {
        "output_dir": "artifacts/training_signal_probe",
        "base": {
            "budget": 1000,
            "allocation_ratios": [0.15],
            "learning_rate": 2e-5,
        },
        "variants": [
            {"name": "a", "learning_rate": 1e-4, "max_steps": 100},
            {"name": "b", "max_steps": 200},
        ],
    }
    runs = build_probe_configs(config)
    assert [name for name, _ in runs] == ["a", "b"]
    assert runs[0][1]["learning_rate"] == 1e-4
    assert runs[1][1]["learning_rate"] == 2e-5
    assert runs[0][1]["output_dir"].endswith("training_signal_probe\\a") or runs[0][1]["output_dir"].endswith(
        "training_signal_probe/a"
    )
    assert runs[0][1]["save_adapter"] is False
    assert runs[0][1]["resume_completed"] is True

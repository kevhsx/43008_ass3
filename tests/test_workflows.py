import json
import sys
from pathlib import Path

import pytest

from warehouse.analysis import analyse
from warehouse.experiment import random_baseline, train_one
from warehouse.provenance import source_hashes, snapshot_sources
from verify_results import verify, verify_sweep


@pytest.fixture
def small_config():
    cfg = json.loads(Path("config.json").read_text())
    cfg["training"].update(episodes=10, seeds=[0], random_episodes_per_seed=2)
    return cfg


@pytest.fixture
def recorded_run(tmp_path, small_config):
    cfg = small_config
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    hashes = source_hashes()
    snapshot_sources(tmp_path, hashes)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "source_sha256": hashes, "requested_methods": ["q_learning", "random"]}))
    train_one(cfg, "q_learning", 0, tmp_path / "q_learning/seed_0")
    random_baseline(cfg, tmp_path)
    analyse(tmp_path)
    return tmp_path


@pytest.mark.parametrize("damage", ["missing", "empty", "duplicate"])
def test_random_baseline_damage_is_rejected(recorded_run, damage):
    path = recorded_run / "random_episodes.csv"
    if damage == "missing":
        path.unlink()
    else:
        rows = path.read_text().splitlines()
        path.write_text(rows[0] + "\n" if damage == "empty" else
                        "\n".join([rows[0], rows[1], rows[1]]) + "\n")
    for check in (analyse, verify):
        with pytest.raises(ValueError, match="[Rr]andom baseline"):
            check(recorded_run)


@pytest.mark.parametrize("damage", ["missing", "extra", "incomplete"])
def test_policy_set_damage_is_rejected(recorded_run, damage):
    folder = recorded_run / "q_learning/seed_0"
    if damage == "missing":
        folder.rename(recorded_run / "hidden_seed")
    elif damage == "extra":
        (recorded_run / "q_learning/seed_99").mkdir()
    else:
        (folder / "training.csv").unlink()
    original = (recorded_run / "summary.json").read_bytes()
    for check in (analyse, verify):
        with pytest.raises(ValueError, match="Missing expected run|Unexpected run|Run is incomplete"):
            check(recorded_run)
    assert (recorded_run / "summary.json").read_bytes() == original


@pytest.mark.parametrize("entry", ["run_sweep", "run_adaptation"])
def test_experiment_entry_rejects_nonempty_output(tmp_path, small_config, monkeypatch, entry):
    import importlib
    output = tmp_path / "recorded"
    output.mkdir()
    sentinel = output / "result.json"
    sentinel.write_text("original record")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(small_config))
    args = [entry, "--output", str(output)]
    if entry == "run_sweep":
        args += ["--config", str(config)]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as exc:
        importlib.import_module(entry).main()
    assert exc.value.code == 2
    assert sentinel.read_text() == "original record"
    assert sorted(p.name for p in output.iterdir()) == ["result.json"]


def test_sweep_records_sources_and_resume_keeps_completed_runs(tmp_path, small_config, monkeypatch):
    import run_sweep
    config = tmp_path / "config.json"
    config.write_text(json.dumps(small_config))
    output = tmp_path / "sweep"
    args = ["run_sweep", "--config", str(config), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    run_sweep.main()
    verify_sweep(output)
    meta = (output / "metadata.json").read_bytes()
    records = {p: p.read_bytes() for p in output.glob("a_*/seed_*/result.json")}
    assert len(records) == 9
    assert (output / "source_snapshot/warehouse/agents.py").exists()
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    run_sweep.main()
    assert (output / "metadata.json").read_bytes() == meta
    assert all(p.read_bytes() == data for p, data in records.items())
    (output / "source_snapshot/warehouse/agents.py").write_text("changed")
    with pytest.raises(AssertionError, match="Source snapshot changed"):
        verify_sweep(output)


def test_adaptation_resume_preserves_run_start_and_records(tmp_path, small_config, monkeypatch):
    import run_adaptation
    from verify_results import verify_adaptation
    base = tmp_path / "main"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(small_config))
    for method in ("q_learning", "dqn"):
        train_one(small_config, method, 0, base / method / "seed_0")
    output = tmp_path / "adaptation"
    args = ["run_adaptation", "--main-results", str(base), "--output", str(output), "--episodes", "10"]
    monkeypatch.setattr(sys, "argv", args)
    run_adaptation.main()
    verify(output)
    verify_adaptation(output, base)
    meta = (output / "metadata.json").read_bytes()
    records = {p: p.read_bytes() for p in output.glob("*/seed_*/result.json")}
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    run_adaptation.main()
    assert (output / "metadata.json").read_bytes() == meta
    assert all(p.read_bytes() == data for p, data in records.items())

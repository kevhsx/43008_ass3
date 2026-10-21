import json
from pathlib import Path
import pytest
from warehouse.experiment import train_one
from warehouse.analysis import analyse
from warehouse.provenance import source_hashes, snapshot_sources
from verify_results import verify


def test_verifier_rejects_missing_expected_policy(tmp_path):
    cfg = json.loads(Path("config.json").read_text())
    cfg["training"].update(episodes=20, seeds=[0, 1])
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    hashes = source_hashes()
    snapshot_sources(tmp_path, hashes)
    (tmp_path / "metadata.json").write_text(json.dumps({"source_sha256": hashes, "requested_methods": ["q_learning"]}))
    for seed in (0, 1):
        train_one(cfg, "q_learning", seed, tmp_path / "q_learning" / f"seed_{seed}")
    analyse(tmp_path)
    verify(tmp_path)
    (tmp_path / "q_learning/seed_1").rename(tmp_path / "hidden_run")
    with pytest.raises(ValueError, match="Missing expected run"):
        verify(tmp_path)


def test_verifier_rejects_tampered_source_snapshot(tmp_path):
    (tmp_path / "config.json").write_text(Path("config.json").read_text())
    (tmp_path / "summary.json").write_text(json.dumps({"trained": {}}))
    hashes = source_hashes(); snapshot_sources(tmp_path, hashes)
    (tmp_path / "metadata.json").write_text(json.dumps({"source_sha256": hashes}))
    (tmp_path / "source_snapshot/warehouse/env.py").write_text("modified")
    with pytest.raises(AssertionError, match="snapshot changed"):
        verify(tmp_path)

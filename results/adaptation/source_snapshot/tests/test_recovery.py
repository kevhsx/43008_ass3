import copy
import json
from pathlib import Path
import numpy as np
import pytest
from warehouse.experiment import train_one
from warehouse.configuration import validate_config
from warehouse.env import WarehouseEnv


def config():
    return json.loads(Path("config.json").read_text())


@pytest.mark.parametrize("method", ["q_learning", "dqn"])
def test_episode_checkpoint_matches_uninterrupted_training(tmp_path, method):
    cfg = config()
    cfg["training"]["episodes"] = 40
    cfg["dqn"].update(learning_starts=32, batch_size=16, buffer_size=500)
    uninterrupted, resumed = tmp_path / "a", tmp_path / "b"
    train_one(cfg, method, 4, uninterrupted, checkpoint_interval=20)
    assert train_one(cfg, method, 4, resumed, checkpoint_interval=20, stop_after=20) is None
    train_one(cfg, method, 4, resumed, checkpoint_interval=20, resume=True)
    assert (uninterrupted / "training.csv").read_text() == (resumed / "training.csv").read_text()
    assert (uninterrupted / "greedy_checks.csv").read_text() == (resumed / "greedy_checks.csv").read_text()
    if method == "q_learning":
        np.testing.assert_array_equal(np.load(uninterrupted / "q_table.npy"), np.load(resumed / "q_table.npy"))
    else:
        import torch
        a, b = [torch.load(p / "model.pt", weights_only=True)["model"] for p in (uninterrupted, resumed)]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_checkpoint_rejects_changed_configuration(tmp_path):
    cfg = config(); cfg["training"]["episodes"] = 40
    train_one(cfg, "q_learning", 0, tmp_path, stop_after=20)
    cfg["q_learning"]["alpha"] = .5
    with pytest.raises(ValueError, match="mismatch"):
        train_one(cfg, "q_learning", 0, tmp_path, resume=True)


@pytest.mark.parametrize("section,key,value", [
    ("training", "evaluation_interval", 0), ("training", "seeds", [0, 0]),
    ("training", "epsilon_min", 2), ("q_learning", "gamma", float("nan")),
    ("dqn", "batch_size", 50000), ("environment", "width", 10.5),
    ("environment", "step_reward", 1), ("environment", "pickup", [4.5, 5]),
])
def test_invalid_configuration_rejected(section, key, value):
    cfg = config();cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_changed_layout_has_valid_twenty_step_route():
    cfg = config();cfg["environment"]["shelves"].extend([[3, 5], [4, 6]])
    env = WarehouseEnv(cfg["environment"])
    assert env.optimal_steps == 20 and env.optimal_return == 23.0

import copy
import json
from pathlib import Path
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from warehouse.env import WarehouseEnv
from warehouse.agents import QLearning, DQN
from warehouse.experiment import evaluate, train_one, convergence


@pytest.fixture
def cfg():
    return json.loads(Path("config.json").read_text())


def follow(env, route):
    total = 0.0
    for a, b in zip(route, route[1:]):
        action = env.MOVES.index((b[0]-a[0], b[1]-a[1]))
        result = env.step(action)
        total += result[1]
    return total, result


@pytest.mark.parametrize("mode", ["discrete", "vector"])
def test_gymnasium_contract(cfg, mode):
    env = WarehouseEnv(cfg["environment"], mode)
    check_env(env, skip_render_check=True)
    assert env.observation_space.contains(env.reset(seed=7)[0])


def test_reset_clears_episode(cfg):
    env = WarehouseEnv(cfg["environment"])
    follow(env, env.shortest_path(env.start, env.pickup))
    env.reset(seed=9)
    assert (env.x, env.y) == env.start
    assert not env.carrying and env.steps == 0 and env.collisions == 0


@pytest.mark.parametrize("action", [1, 2])
def test_boundary_collision(cfg, action):
    env = WarehouseEnv(cfg["environment"])
    _, reward, terminated, truncated, info = env.step(action)
    assert info["position"] == env.start and info["collision"]
    assert info["collisions"] == 1 and reward == pytest.approx(-1.1)
    assert not terminated and not truncated


def test_shelf_collision(cfg):
    env = WarehouseEnv(cfg["environment"])
    follow(env, env.shortest_path(env.start, (1, 7)))
    _, reward, _, _, info = env.step(3)
    assert info["position"] == (1, 7) and info["collision"]
    assert reward == pytest.approx(-1.1)


def test_pickup_reward_once(cfg):
    env = WarehouseEnv(cfg["environment"])
    _, result = follow(env, env.shortest_path(env.start, env.pickup))
    assert result[1] == pytest.approx(4.9) and result[4]["carrying"]
    env.step(0)
    _, reward, _, _, info = env.step(1)
    assert reward == pytest.approx(-.1) and info["carrying"]


def test_delivery_requires_item(cfg):
    env = WarehouseEnv(cfg["environment"])
    route = [(0, y) for y in range(9, -1, -1)] + [(x, 0) for x in range(1, 10)]
    _, result = follow(env, route)
    assert not result[2] and not result[4]["success"] and result[1] == pytest.approx(-.1)


def test_bfs_delivery_reward_and_end(cfg):
    env = WarehouseEnv(cfg["environment"])
    total, result = follow(env, env.optimal_route)
    assert env.optimal_steps == 18 and total == pytest.approx(23.2)
    assert result[1] == pytest.approx(19.9) and result[2] and not result[3]
    with pytest.raises(RuntimeError):
        env.step(0)


def test_time_limit(cfg):
    env = WarehouseEnv(cfg["environment"])
    for _ in range(200):
        result = env.step(2)
    assert not result[2] and result[3] and result[4]["steps"] == 200


def test_delivery_at_cap_is_terminal(cfg):
    cfg["environment"]["max_steps"] = 18
    env = WarehouseEnv(cfg["environment"])
    _, result = follow(env, env.optimal_route)
    assert result[2] and not result[3]


def test_state_carries_markov_information(cfg):
    env = WarehouseEnv(cfg["environment"])
    empty = env.observation()
    env.carrying = True
    assert env.observation() == empty + 100
    vector = WarehouseEnv(cfg["environment"], "vector")
    assert vector.observation().tolist() == [0.0, 1.0, 0.0]


def test_invalid_actions(cfg):
    env = WarehouseEnv(cfg["environment"])
    for action in (-1, 4, 1.5):
        with pytest.raises(ValueError):
            env.step(action)


def test_disconnected_layout_rejected(cfg):
    cfg["environment"]["shelves"] = [[x, 8] for x in range(10)]
    with pytest.raises(ValueError, match="not connected"):
        WarehouseEnv(cfg["environment"])


def test_q_targets_mask_only_termination(cfg):
    agent = QLearning(200, cfg["q_learning"])
    agent.q[1, :] = 10
    agent.update(0, 0, 2, 1, False)
    assert agent.q[0, 0] == pytest.approx(.2 * (2+.95*10))
    agent.update(2, 0, 2, 1, True)
    assert agent.q[2, 0] == pytest.approx(.2*2)


@pytest.mark.parametrize("terminated,expected_loss", [(True, 2.5), (False, 3.5)])
def test_dqn_terminal_and_truncated_targets(cfg, terminated, expected_loss):
    import torch
    settings = copy.deepcopy(cfg["dqn"])
    settings.update(batch_size=1, learning_starts=1, train_frequency=1, learning_rate=0.0, gamma=.5)
    agent = DQN(settings, 0)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        for p in agent.target.parameters():
            p.zero_()
        agent.target[-1].bias.fill_(2)
    loss = agent.update(np.zeros(3, np.float32), 0, 3, np.ones(3, np.float32), terminated)
    assert loss == pytest.approx(expected_loss)


def test_seed_reproducibility_and_isolated_greedy_checks(cfg, tmp_path):
    cfg["training"]["episodes"] = 100
    train_one(cfg, "q_learning", 1, tmp_path / "a")
    train_one(cfg, "q_learning", 1, tmp_path / "b")
    np.testing.assert_array_equal(np.load(tmp_path / "a/q_table.npy"), np.load(tmp_path / "b/q_table.npy"))
    cfg["training"]["evaluation_interval"] = 100
    train_one(cfg, "q_learning", 1, tmp_path / "c")
    np.testing.assert_array_equal(np.load(tmp_path / "a/q_table.npy"), np.load(tmp_path / "c/q_table.npy"))


def test_final_policy_learns_and_checkpoint_roundtrip(cfg, tmp_path):
    cfg["training"]["episodes"] = 300
    record = train_one(cfg, "q_learning", 0, tmp_path)
    assert record["optimal"] and record["steps"] == 18
    agent = QLearning(200, cfg["q_learning"])
    agent.q = np.load(tmp_path / "q_table.npy")
    assert evaluate(WarehouseEnv(cfg["environment"]), agent)["route"] == record["route"]


def test_dqn_checkpoint_replay(cfg, tmp_path):
    import torch
    agent = DQN(cfg["dqn"], 1)
    state = np.array([.2, .3, 1], dtype=np.float32)
    agent.save(tmp_path / "model.pt")
    other = DQN(cfg["dqn"], 7)
    other.load(tmp_path / "model.pt")
    with torch.no_grad():
        torch.testing.assert_close(agent.online(torch.from_numpy(state)), other.online(torch.from_numpy(state)))


def test_convergence_censors_failure_and_late_regression():
    checks = [{"episode": e, "success": 1, "steps": s} for e, s in [(10, 18), (20, 20), (30, 18)]]
    assert convergence(checks, 18) == (10, 30)
    checks.append({"episode": 40, "success": 0, "steps": 200})
    assert convergence(checks, 18) == (10, None)


def test_render_modes(cfg):
    rgb = WarehouseEnv(cfg["environment"], render_mode="rgb_array").render()
    assert rgb.shape == (400, 400, 3) and rgb.dtype == np.uint8
    ansi = WarehouseEnv(cfg["environment"], render_mode="ansi").render()
    assert "#" in ansi and "P" in ansi and len(ansi.splitlines()) == 10

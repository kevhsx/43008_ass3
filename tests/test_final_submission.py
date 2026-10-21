import copy
import json
import sys
from pathlib import Path

import pytest

from run_sweep import main as run_sweep
from verify_results import verify, verify_sweep
from warehouse.analysis import analyse, read_csv, validate_greedy_checks
from warehouse.configuration import epsilon_decay_steps
from warehouse.experiment import train_one, write_csv
from warehouse.provenance import source_hashes, snapshot_sources


@pytest.mark.parametrize("start,minimum,decay,expected", [
    (1, .05, .99, 299), (1, .05, .995, 598), (1, .05, .999, 2995),
    (1, 0, .995, None), (0, 0, .995, 0), (.05, .05, .995, 0),
    (1, .05, 1, None), (1, 1, 1, 0),
])
def test_exploration_floor_boundaries(start, minimum, decay, expected):
    assert epsilon_decay_steps(start, minimum, decay) == expected


def test_greedy_checks_include_final_noninterval_episode():
    cfg = {"training": {"episodes": 17, "evaluation_interval": 5}}
    logs = [{"seed": 0, "episode": i, "steps": 2, "cumulative_steps": i * 2}
            for i in range(1, 18)]
    checks = [dict(logs[i - 1]) for i in (5, 10, 15, 17)]
    validate_greedy_checks(checks, cfg, 0, logs)
    cfg["training"]["evaluation_interval"] = 100
    validate_greedy_checks([checks[-1]], cfg, 0, logs)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "reordered", "wrong_seed", "wrong_steps", "missing_final"])
def test_invalid_greedy_checks_rejected_before_report_write(tmp_path, damage):
    cfg = json.loads(Path("config.json").read_text())
    cfg["training"].update(episodes=25, seeds=[0])
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    hashes = source_hashes()
    snapshot_sources(tmp_path, hashes)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "source_sha256": hashes, "requested_methods": ["q_learning"]}))
    folder = tmp_path / "q_learning/seed_0"
    train_one(cfg, "q_learning", 0, folder)
    analyse(tmp_path)
    original = (tmp_path / "summary.json").read_bytes()
    checks = read_csv(folder / "greedy_checks.csv")
    if damage == "missing":
        checks.pop(1)
    elif damage == "missing_final":
        checks.pop()
    elif damage == "duplicate":
        checks.insert(1, copy.deepcopy(checks[0]))
    elif damage == "reordered":
        checks[0], checks[1] = checks[1], checks[0]
    elif damage == "wrong_seed":
        checks[0]["seed"] = 99
    else:
        checks[0]["cumulative_steps"] = 0
    write_csv(folder / "greedy_checks.csv", checks)
    for check in (analyse, verify):
        with pytest.raises(ValueError, match="Greedy-check"):
            check(tmp_path)
    assert (tmp_path / "summary.json").read_bytes() == original


@pytest.mark.parametrize("start,minimum,expected", [(1, 0, ""), (0, 0, "0"), (.05, .05, "0")])
def test_zero_floor_sweep_completes_and_verifies(tmp_path, monkeypatch, start, minimum, expected):
    cfg = json.loads(Path("config.json").read_text())
    cfg["training"].update(episodes=11, seeds=[0], epsilon_start=start, epsilon_min=minimum)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    out = tmp_path / "sweep"
    monkeypatch.setattr(sys, "argv", ["run_sweep", "--config", str(config), "--output", str(out)])
    run_sweep()
    rows = read_csv(out / "sweep_summary.csv")
    assert len(rows) == 9 and all(row["decays_to_min"] == expected for row in rows)
    verify_sweep(out)
    path = out / "a_0.05_decay_0.99/seed_0/greedy_checks.csv"
    checks = read_csv(path)
    write_csv(path, checks[1:])
    with pytest.raises(ValueError, match="Greedy-check"):
        verify_sweep(out)


def test_manual_keyboard_pause_reset_and_exit(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    import demo
    import warehouse.env

    instances = []
    original_env = warehouse.env.WarehouseEnv

    class ObservedEnv(original_env):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(warehouse.env, "WarehouseEnv", ObservedEnv)
    frames = iter([
        [pygame.K_UP],
        [pygame.K_SPACE, pygame.K_UP],
        [pygame.K_SPACE, pygame.K_UP],
        [pygame.K_r],
        [pygame.K_ESCAPE],
    ])
    steps = []

    def next_events():
        steps.append(instances[0].steps)
        return [pygame.event.Event(pygame.KEYDOWN, key=key) for key in next(frames)]

    monkeypatch.setattr(pygame.event, "get", next_events)
    monkeypatch.setattr(sys, "argv", ["demo", "--method", "manual", "--fps", "1000"])
    try:
        demo.main()
    finally:
        pygame.quit()
    assert steps == [0, 1, 1, 2, 0]
    assert (instances[0].x, instances[0].y) == instances[0].start


def test_report_export_rejects_missing_checks_before_creating_figures(tmp_path, monkeypatch):
    import shutil
    from tools import export_report_figures

    main = tmp_path / "results/main"
    shutil.copytree("results/main", main)
    path = main / "q_learning/seed_0/greedy_checks.csv"
    rows = read_csv(path)
    rows.pop(len(rows) // 2)
    write_csv(path, rows)
    monkeypatch.setattr(export_report_figures, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Greedy-check"):
        export_report_figures.export()
    assert not (tmp_path / "output/figures").exists()

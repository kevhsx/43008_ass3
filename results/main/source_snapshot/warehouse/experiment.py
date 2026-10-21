"""Seeded experiments with raw logs and final-checkpoint evaluation."""
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
from .env import WarehouseEnv
from .agents import QLearning, DQN


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(env, agent=None, rng=None):
    state, _ = env.reset()
    total, route = 0.0, [env.start]
    while True:
        action = int(rng.integers(4)) if agent is None else agent.greedy(state)
        state, reward, terminated, truncated, info = env.step(action)
        total += reward
        route.append(info["position"])
        if terminated or truncated:
            return {"success": int(info["success"]), "return": total, "steps": info["steps"],
                    "collisions": info["collisions"],
                    "collision_rate": info["collisions"] / info["steps"], "route": route}


def convergence(checks, optimum):
    first = next((r["episode"] for r in checks if r["success"]), None)
    stable = None
    # Stable through the OBSERVED training horizon, not a future guarantee.
    for r in reversed(checks):
        if r["success"] and r["steps"] == optimum:
            stable = r["episode"]
        else:
            break
    return first, stable


def train_one(cfg, method, seed, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mode = "vector" if method == "dqn" else "discrete"
    env, eval_env = WarehouseEnv(cfg["environment"], mode), WarehouseEnv(cfg["environment"], mode)
    rng = np.random.default_rng(seed)
    agent = (DQN(cfg["dqn"], seed) if method == "dqn" else
             QLearning(env.observation_space.n, cfg["q_learning"]))
    schedule = cfg["training"]
    epsilon, cumulative_steps = schedule["epsilon_start"], 0
    logs, checks, eval_seconds = [], [], 0.0
    start = time.perf_counter()
    for episode in range(1, schedule["episodes"] + 1):
        state, _ = env.reset(seed=seed if episode == 1 else None)
        total, losses = 0.0, []
        while True:
            action = int(rng.integers(4)) if rng.random() < epsilon else agent.greedy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            loss = agent.update(state, action, reward, next_state, terminated)
            if loss is not None:
                losses.append(loss)
            state, total = next_state, total + reward
            if terminated or truncated:
                break
        cumulative_steps += info["steps"]
        logs.append({"seed": seed, "episode": episode, "epsilon": epsilon,
                     "return": total, "success": int(info["success"]), "steps": info["steps"],
                     "collisions": info["collisions"], "collision_rate": info["collisions"] / info["steps"],
                     "cumulative_steps": cumulative_steps,
                     "mean_loss": float(np.mean(losses)) if losses else ""})
        if episode % schedule["evaluation_interval"] == 0 or episode == schedule["episodes"]:
            tick = time.perf_counter()
            row = evaluate(eval_env, agent)
            row.pop("route")
            checks.append({"seed": seed, "episode": episode, "cumulative_steps": cumulative_steps, **row})
            eval_seconds += time.perf_counter() - tick
        epsilon = max(schedule["epsilon_min"], epsilon * schedule["epsilon_decay"])
    elapsed = time.perf_counter() - start
    final = evaluate(eval_env, agent)
    first, stable = convergence(checks, env.optimal_steps)
    record = {"method": method, "seed": seed, **final,
              "optimal": bool(final["success"] and final["steps"] == env.optimal_steps),
              "first_success_check": first, "stable_optimal_check": stable,
              "steps_to_stable_optimal": next((r["cumulative_steps"] for r in checks if r["episode"] == stable), None),
              "environment_steps": cumulative_steps, "training_seconds": elapsed - eval_seconds,
              "evaluation_seconds": eval_seconds, "training_with_checks_seconds": elapsed}
    write_csv(out / "training.csv", logs)
    write_csv(out / "greedy_checks.csv", checks)
    (out / "result.json").write_text(json.dumps(record, indent=2))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    agent.save(out / ("model.pt" if method == "dqn" else "q_table.npy"))
    print(f"{method} seed={seed}: success={final['success']} steps={final['steps']} "
          f"stable={stable} train={record['training_seconds']:.2f}s", flush=True)
    return record


def random_baseline(cfg, out):
    env, rows = WarehouseEnv(cfg["environment"]), []
    for seed in cfg["training"]["seeds"]:
        rng = np.random.default_rng(seed)
        for episode in range(1, cfg["training"]["random_episodes_per_seed"] + 1):
            row = evaluate(env, rng=rng)
            row.pop("route")
            rows.append({"seed": seed, "episode": episode, **row})
    write_csv(Path(out) / "random_episodes.csv", rows)
    return rows


def metadata():
    import importlib.metadata
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "device": "CPU", "torch_threads": 1,
            "versions": {m: importlib.metadata.version(m) for m in
                         ("numpy", "gymnasium", "torch", "matplotlib", "pygame", "pytest")},
            "timing": "Training loop including logging; excludes greedy checks, model creation, final evaluation and disk writes",
            "time_limit": "External rollout cap; bootstrap on truncation, mask on delivery termination only"}

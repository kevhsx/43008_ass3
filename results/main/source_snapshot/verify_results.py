"""Cross-check raw logs, final saved policies, metrics and training provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from warehouse.env import WarehouseEnv
from warehouse.agents import QLearning, DQN
from warehouse.experiment import evaluate, convergence
from warehouse.analysis import read_csv, stats


def verify(out):
    out = Path(out)
    cfg = json.loads((out / "config.json").read_text())
    meta = json.loads((out / "metadata.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    for name, expected in meta["source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, f"Source changed: {name}"
    count, policies = 0, {}
    for method in ("q_learning", "dqn"):
        records = []
        for seed in cfg["training"]["seeds"]:
            folder = out / method / f"seed_{seed}"
            if not folder.exists():
                continue
            if not (folder / "result.json").exists():
                raise ValueError(f"Run is incomplete: {folder}. Wait for training to finish before verification.")
            record = json.loads((folder / "result.json").read_text())
            assert json.loads((folder / "config.json").read_text()) == cfg
            logs = read_csv(folder / "training.csv")
            assert len(logs) == cfg["training"]["episodes"]
            assert [int(r["episode"]) for r in logs] == list(range(1, len(logs)+1))
            cumulative = 0
            for row in logs:
                steps, collisions = int(row["steps"]), int(row["collisions"])
                cumulative += steps
                assert int(row["cumulative_steps"]) == cumulative
                assert 0 <= collisions <= steps <= cfg["environment"]["max_steps"]
                assert int(row["success"]) or steps == cfg["environment"]["max_steps"]
                assert np.isclose(float(row["collision_rate"]), collisions/steps)
                if row["mean_loss"]:
                    assert np.isfinite(float(row["mean_loss"]))
            assert record["environment_steps"] == cumulative
            checks = read_csv(folder / "greedy_checks.csv")
            typed = [{**r, "episode": int(r["episode"]), "steps": int(r["steps"]), "success": int(r["success"])} for r in checks]
            first, stable = convergence(typed, summary["optimal_steps"])
            assert (record["first_success_check"], record["stable_optimal_check"]) == (first, stable)
            mode = "vector" if method == "dqn" else "discrete"
            env = WarehouseEnv(cfg["environment"], mode)
            if method == "q_learning":
                agent = QLearning(env.observation_space.n, cfg["q_learning"])
                agent.q = np.load(folder / "q_table.npy")
            else:
                agent = DQN(cfg["dqn"], seed)
                agent.load(folder / "model.pt")
            actual = evaluate(env, agent)
            for key in ("success", "steps", "collisions", "return", "collision_rate"):
                assert np.isclose(actual[key], record[key]), f"Saved model differs: {method}/{seed}/{key}"
            assert [list(p) for p in actual["route"]] == record["route"]
            assert bool(record["optimal"]) == bool(actual["success"] and actual["steps"] == env.optimal_steps)
            records.append(record); count += 1
        if records:
            if method not in summary["trained"]:
                raise ValueError(f"Summary does not yet include {method}; rerun analysis after training finishes.")
            s = summary["trained"][method]
            assert s["n_policies"] == len(records)
            assert s["successful_policies"] == sum(r["success"] for r in records)
            assert s["optimal_policies"] == sum(r["optimal"] for r in records)
            for key in ("return", "steps", "collision_rate", "environment_steps", "training_seconds",
                        "first_success_check", "stable_optimal_check", "steps_to_stable_optimal"):
                assert s[key] == stats([r[key] for r in records if r[key] is not None])
            policies[method] = len(records)
    random_rows = read_csv(out / "random_episodes.csv") if (out / "random_episodes.csv").exists() else []
    if random_rows:
        s = summary["random"]
        assert len(random_rows) == len(cfg["training"]["seeds"]) * cfg["training"]["random_episodes_per_seed"]
        assert s["n_episodes"] == len(random_rows)
        assert s["successes"] == sum(int(r["success"]) for r in random_rows)
        for key in ("return", "steps", "collision_rate"):
            assert s[key] == stats([float(r[key]) for r in random_rows])
        env = WarehouseEnv(cfg["environment"])
        for seed in cfg["training"]["seeds"]:
            rng = np.random.default_rng(seed)
            seeded_rows = [r for r in random_rows if int(r["seed"]) == seed]
            for row in seeded_rows:
                actual = evaluate(env, rng=rng)
                for key in ("success", "return", "steps", "collisions", "collision_rate"):
                    assert np.isclose(actual[key], float(row[key])), f"Random replay differs: seed={seed}, episode={row['episode']}"
    record = {"passed": True, "verified_policies": count, "policies_by_method": policies,
              "random_episodes": len(random_rows), "checks": "source hashes, configuration, raw episode logs, convergence, saved-policy replay, random-baseline replay and summary statistics"}
    (out / "verification.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


def verify_sweep(out):
    out = Path(out)
    cfg = json.loads((out / "base_config.json").read_text())
    rows = read_csv(out / "sweep_summary.csv")
    assert len(rows) == 9
    runs = 0
    for row in rows:
        records = []
        for seed in cfg["training"]["seeds"]:
            folder = out / f"a_{row['alpha']}_decay_{row['epsilon_decay']}" / f"seed_{seed}"
            r = json.loads((folder / "result.json").read_text())
            checks = read_csv(folder / "greedy_checks.csv")
            typed = [dict(c, episode=int(c["episode"]), success=int(c["success"]), steps=int(c["steps"])) for c in checks]
            env = WarehouseEnv(cfg["environment"])
            assert r["stable_optimal_check"] == convergence(typed, env.optimal_steps)[1]
            agent = QLearning(env.observation_space.n, cfg["q_learning"])
            agent.q = np.load(folder / "q_table.npy")
            actual = evaluate(env, agent)
            assert actual["steps"] == r["steps"] and actual["success"] == r["success"]
            assert np.isclose(actual["return"], r["return"])
            records.append(r); runs += 1
        stable = stats([r["stable_optimal_check"] for r in records if r["stable_optimal_check"] is not None])
        steps = stats([r["environment_steps"] for r in records])
        assert int(row["optimal_policies"]) == sum(r["optimal"] for r in records)
        assert float(row["stable_mean"]) == stable["mean"]
        assert float(row["stable_std"]) == stable["std"]
        assert float(row["environment_steps_mean"]) == steps["mean"]
        assert float(row["environment_steps_std"]) == steps["std"]
    result = {"passed": True, "settings": len(rows), "runs": runs,
              "checks": "raw greedy checks, convergence, environment-step summaries and final saved-policy replay"}
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/main")
    parser.add_argument("--sweep", help="Also verify a Q-Learning sweep directory")
    try:
        args = parser.parse_args()
        verify(args.results)
        if args.sweep:
            verify_sweep(args.sweep)
    except (ValueError, AssertionError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")

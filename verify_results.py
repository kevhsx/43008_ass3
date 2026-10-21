"""Check saved models, logs, statistics and source files."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from warehouse.env import WarehouseEnv
from warehouse.agents import QLearning, DQN
from warehouse.experiment import evaluate, convergence
from warehouse.analysis import read_csv, stats, validate_random_rows, validate_greedy_checks
from warehouse.configuration import epsilon_decay_steps
from warehouse.provenance import recorded_run_folders


def verify(out):
    out = Path(out)
    cfg = json.loads((out / "config.json").read_text())
    meta = json.loads((out / "metadata.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    snapshot_checked = 0
    for name, expected in meta["source_sha256"].items():
        snapshot = out / "source_snapshot" / name
        path = snapshot if snapshot.exists() else Path(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Source snapshot changed: {name}"
        snapshot_checked += int(snapshot.exists())
    count, policies = 0, {}
    for method in ("q_learning", "dqn"):
        if method not in summary["trained"]:
            if method in meta.get("requested_methods", []):
                raise ValueError(f"Requested method {method} has not completed")
            continue
        records = []
        recorded_run_folders(out, method, cfg["training"]["seeds"])
        for seed in cfg["training"]["seeds"]:
            folder = out / method / f"seed_{seed}"
            record = json.loads((folder / "result.json").read_text())
            assert record["seed"] == seed and record["method"] == method
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
            validate_greedy_checks(checks, cfg, seed, logs)
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
    random_path = out / "random_episodes.csv"
    if ("random" in meta.get("requested_methods", []) or summary.get("random") is not None) and not random_path.is_file():
        raise ValueError("Missing requested random baseline: random_episodes.csv")
    random_rows = read_csv(random_path) if random_path.exists() else []
    if random_path.exists():
        validate_random_rows(random_rows, cfg)
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
            for row in sorted(seeded_rows, key=lambda row: int(row["episode"])):
                actual = evaluate(env, rng=rng)
                for key in ("success", "return", "steps", "collisions", "collision_rate"):
                    assert np.isclose(actual[key], float(row[key])), f"Random replay differs: seed={seed}, episode={row['episode']}"
    record = {"passed": True, "verified_policies": count, "policies_by_method": policies,
              "source_snapshot_files_checked": snapshot_checked,
              "random_episodes": len(random_rows), "checks": "source hashes, configuration, raw episode logs, convergence, saved-policy replay, random-baseline replay and summary statistics"}
    (out / "verification.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


def verify_sweep(out):
    out = Path(out)
    cfg = json.loads((out / "base_config.json").read_text())
    meta_path = out / "metadata.json"
    snapshots_checked = 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for name, expected in meta["source_sha256"].items():
            snapshot = out / "source_snapshot" / name
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected, f"Source snapshot changed: {name}"
            snapshots_checked += 1
    rows = read_csv(out / "sweep_summary.csv")
    expected_settings = {(alpha, decay) for alpha in (.05, .2, .5) for decay in (.99, .995, .999)}
    assert len(rows) == 9 and {(float(r["alpha"]), float(r["epsilon_decay"])) for r in rows} == expected_settings
    runs = 0
    for row in rows:
        records = []
        expected_decays = epsilon_decay_steps(cfg["training"]["epsilon_start"], cfg["training"]["epsilon_min"], float(row["epsilon_decay"]))
        if (int(row["decays_to_min"]) if row["decays_to_min"] else None) != expected_decays:
            raise ValueError("Sweep exploration decay summary does not match configuration")
        parent = f"a_{row['alpha']}_decay_{row['epsilon_decay']}"
        recorded_run_folders(out, parent, cfg["training"]["seeds"])
        for seed in cfg["training"]["seeds"]:
            folder = out / f"a_{row['alpha']}_decay_{row['epsilon_decay']}" / f"seed_{seed}"
            r = json.loads((folder / "result.json").read_text())
            run_cfg = json.loads((folder / "config.json").read_text())
            assert run_cfg["q_learning"]["alpha"] == float(row["alpha"])
            assert run_cfg["training"]["epsilon_decay"] == float(row["epsilon_decay"])
            assert r["seed"] == seed and r["method"] == "q_learning"
            checks = read_csv(folder / "greedy_checks.csv")
            validate_greedy_checks(checks, run_cfg, seed, read_csv(folder / "training.csv"))
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
        for column, key in (("stable_mean", "mean"), ("stable_std", "std")):
            assert (float(row[column]) if row[column] else None) == stable[key]
        assert float(row["environment_steps_mean"]) == steps["mean"]
        assert (float(row["environment_steps_std"]) if row["environment_steps_std"] else None) == steps["std"]
    result = {"passed": True, "settings": len(rows), "runs": runs,
              "source_snapshot_files_checked": snapshots_checked,
              "source_provenance": "verified" if meta_path.exists() else "historical run has no sweep metadata; source hashes unavailable",
              "checks": "raw greedy checks, convergence, environment-step summaries and final saved-policy replay"}
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def verify_adaptation(out, base_results=None):
    out = Path(out)
    cfg = json.loads((out / "config.json").read_text())
    meta = json.loads((out / "metadata.json").read_text())
    base = Path(base_results or meta["base_results"])
    if not base.exists():
        base = out.parent / "main"
    from run_adaptation import load_policy
    frozen = json.loads((out / "frozen_results.json").read_text())
    expected = {(method, seed) for method in ("q_learning", "dqn") for seed in cfg["training"]["seeds"]}
    actual = [(record["method"], record["seed"]) for record in frozen]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Frozen evaluation must contain each configured method/seed exactly once")
    comparison = json.loads((out / "adaptation_summary.json").read_text())
    for record in frozen:
        method, seed = record["method"], record["seed"]
        model = base / method / f"seed_{seed}" / ("q_table.npy" if method == "q_learning" else "model.pt")
        assert hashlib.sha256(model.read_bytes()).hexdigest() == meta["base_models"][f"{method}/seed_{seed}"]["sha256"]
        agent = load_policy(cfg, method, seed, model)
        mode = "discrete" if method == "q_learning" else "vector"
        actual = evaluate(WarehouseEnv(cfg["environment"], mode), agent)
        for key in ("success", "return", "steps", "collisions"):
            assert np.isclose(actual[key], record[key])
    for method in ("q_learning", "dqn"):
        rows = [r for r in frozen if r["method"] == method]
        assert len(rows) == len(cfg["training"]["seeds"])
        assert comparison[method]["frozen"]["successful_policies"] == sum(r["success"] for r in rows)
    result = {"passed": True, "frozen_policies_replayed": len(frozen),
              "checks": "base model hashes, frozen replay and changed-layout success counts"}
    (out / "frozen_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/main")
    parser.add_argument("--sweep", help="Also verify a Q-Learning sweep directory")
    parser.add_argument("--adaptation", help="Also replay the frozen-policy adaptation stage")
    parser.add_argument("--base-results", help="Override main-results path for relocated adaptation data")
    try:
        args = parser.parse_args()
        verify(args.results)
        if args.sweep:
            verify_sweep(args.sweep)
        if args.adaptation:
            verify_adaptation(args.adaptation, args.base_results)
    except (ValueError, AssertionError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")

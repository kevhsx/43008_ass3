"""Test saved policies on a changed map, then train them further."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from warehouse.env import WarehouseEnv
from warehouse.agents import QLearning, DQN
from warehouse.experiment import evaluate, train_one, metadata, write_csv
from warehouse.analysis import analyse, stats, fmt
from warehouse.configuration import validate_config
from warehouse.provenance import guard_output_directory, source_hashes, snapshot_sources


def load_policy(cfg, method, seed, model):
    if method == "q_learning":
        agent = QLearning(cfg["environment"]["width"] * cfg["environment"]["height"] * 2, cfg[method])
        agent.q = np.load(model, allow_pickle=False)
    else:
        agent = DQN(cfg[method], seed)
        agent.load(model)
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main-results", default="results/main")
    p.add_argument("--output", default="results/adaptation")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    base, out = Path(args.main_results), Path(args.output)
    try:
        guard_output_directory(out, args.resume)
    except ValueError as exc:
        p.error(str(exc))
    cfg = json.loads((base / "config.json").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["environment"]["shelves"].extend([[3, 5], [4, 6]])
    cfg["training"]["episodes"] = args.episodes
    validate_config(cfg)
    env = WarehouseEnv(cfg["environment"])
    if env.optimal_steps != 20:
        p.error("This extension is defined for the supplied main layout; expected 20-step stress-test optimum")
    models = {}
    for method in ("q_learning", "dqn"):
        for seed in cfg["training"]["seeds"]:
            file = base / method / f"seed_{seed}" / ("q_table.npy" if method == "q_learning" else "model.pt")
            if not file.exists():
                p.error(f"Missing base model: {file}")
            models[f"{method}/seed_{seed}"] = {"path": str(file.resolve()), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != cfg:
        p.error("Existing adaptation configuration differs; choose a new output")
    hashes = source_hashes()
    existing = {}
    if (out / "metadata.json").exists():
        existing = json.loads((out / "metadata.json").read_text())
        if existing["source_sha256"] != hashes or existing["base_models"] != models:
            p.error("Adaptation sources or initial models changed; choose a new output")
    for mode in ("discrete", "vector"):
        checked = gym.make("WarehouseNav-v0", settings=cfg["environment"], observation_mode=mode)
        check_env(checked.unwrapped)
        checked.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    snapshot_sources(out, hashes)
    (out / "metadata.json").write_text(json.dumps({**metadata(), "started_utc": existing.get("started_utc", datetime.now(timezone.utc).isoformat()),
         "source_sha256": hashes, "requested_methods": ["q_learning", "dqn"], "base_models": models,
         "base_results": str(base.resolve()), "added_shelves": [[3, 5], [4, 6]],
         "check_env": "passed (registered environment, both observations)",
         "protocol": "All base final checkpoints; no changed-map tuning. Frozen evaluation first. Fresh epsilon schedule and replay/Adam for warm-start adaptation. Extra budget reported separately."}, indent=2))
    frozen = []
    for method in ("q_learning", "dqn"):
        mode = "discrete" if method == "q_learning" else "vector"
        for seed in cfg["training"]["seeds"]:
            model = Path(models[f"{method}/seed_{seed}"]["path"])
            agent = load_policy(cfg, method, seed, model)
            record = evaluate(WarehouseEnv(cfg["environment"], mode), agent)
            frozen.append({"method": method, "seed": seed, **record,
                           "optimal": bool(record["success"] and record["steps"] == env.optimal_steps)})
            folder = out / method / f"seed_{seed}"
            if args.resume and (folder / "result.json").exists():
                if json.loads((folder / "config.json").read_text()) != cfg:
                    p.error(f"Per-run configuration mismatch: {folder}")
            else:
                train_one(cfg, method, seed, folder, resume=args.resume, initial_model=model)
    (out / "frozen_results.json").write_text(json.dumps(frozen, indent=2))
    write_csv(out / "frozen_policies.csv", [{k: v for k, v in r.items() if k != "route"} for r in frozen])
    analyse(out)
    summary = json.loads((out / "summary.json").read_text())
    comparison = {}
    lines = ["# Changed-layout transfer and adaptation", "", "Two additional shelves at (3,5) and (4,6) block the original approaches to pickup. BFS optimum: 20 moves; undiscounted return 23.0.",
             "", "This is a deliberately route-blocking stress test, not a representative random sample of warehouses. The new layout was excluded from the original main experiment and was not used to retune either algorithm.",
             "", f"Each of the ten base final checkpoints per method was evaluated frozen, then warm-started for {args.episodes} additional episodes. DQN weights are retained; replay and Adam start fresh. Exploration restarts at 1.0 with the original decay.", "",
             "| Method | Frozen successful policies | Adapted successful policies | Adapted optimal policies | Adapted return | Additional environment steps | Additional training seconds |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ("q_learning", "dqn"):
        f = [r for r in frozen if r["method"] == method]
        a = summary["trained"][method]
        fs = {"successful_policies": sum(r["success"] for r in f), "n_policies": len(f),
              "return": stats([r["return"] for r in f]), "steps": stats([r["steps"] for r in f]),
              "collision_rate": sum(r["collisions"] for r in f)/sum(r["steps"] for r in f)}
        comparison[method] = {"frozen": fs, "adapted": a}
        lines.append(f"| {method} | {fs['successful_policies']}/{len(f)} | {a['successful_policies']}/{a['n_policies']} | {a['optimal_policies']}/{a['n_policies']} | {fmt(a['return'])} | {fmt(a['environment_steps'], 0)} | {fmt(a['training_seconds'], 2)} |")
    lines += ["", "Adapted-policy metrics do not replace the main-layout metrics. A successful adaptation demonstrates relearning after new interaction, not zero-shot generalisation or faster learning than training from scratch. No from-scratch comparator or diverse map suite is included."]
    (out / "ADAPTATION.md").write_text("\n".join(lines)+"\n")
    (out / "adaptation_summary.json").write_text(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

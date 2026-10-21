"""Q-Learning robustness sweep; final settings are not selected using this sweep."""
import argparse
import copy
import json
import math
from pathlib import Path
from warehouse.experiment import train_one, write_csv
from warehouse.analysis import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="results/sweep")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    saved = out / "base_config.json"
    if saved.exists() and json.loads(saved.read_text()) != cfg:
        parser.error("Different existing configuration; choose a new output")
    saved.write_text(json.dumps(cfg, indent=2))
    rows = []
    for alpha in (.05, .2, .5):
        for decay in (.99, .995, .999):
            run_cfg = copy.deepcopy(cfg)
            run_cfg["q_learning"]["alpha"] = alpha
            run_cfg["training"]["epsilon_decay"] = decay
            records = [train_one(run_cfg, "q_learning", seed, out / f"a_{alpha}_decay_{decay}" / f"seed_{seed}") for seed in cfg["training"]["seeds"]]
            stable = stats([r["stable_optimal_check"] for r in records if r["stable_optimal_check"] is not None])
            steps = stats([r["environment_steps"] for r in records])
            rows.append({"alpha": alpha, "epsilon_decay": decay,
                         "decays_to_min": math.ceil(math.log(cfg["training"]["epsilon_min"]/cfg["training"]["epsilon_start"])/math.log(decay)),
                         "optimal_policies": sum(r["optimal"] for r in records), "n_runs": len(records),
                         "stable_n": stable["n"], "stable_mean": stable["mean"], "stable_std": stable["std"],
                         "environment_steps_mean": steps["mean"], "environment_steps_std": steps["std"]})
    write_csv(out / "sweep_summary.csv", rows)


if __name__ == "__main__":
    main()

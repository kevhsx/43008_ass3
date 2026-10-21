"""Compare Q-Learning settings without changing the main experiment."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from warehouse.experiment import train_one, write_csv, metadata
from warehouse.analysis import stats
from warehouse.configuration import validate_config, epsilon_decay_steps
from warehouse.provenance import guard_output_directory, source_hashes, snapshot_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="results/sweep")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    validate_config(cfg)
    out = Path(args.output)
    try:
        guard_output_directory(out, args.resume, config_name="base_config.json")
    except ValueError as exc:
        parser.error(str(exc))
    saved = out / "base_config.json"
    if saved.exists() and json.loads(saved.read_text()) != cfg:
        parser.error("Different existing configuration; choose a new output")
    hashes = source_hashes()
    meta_path = out / "metadata.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if existing and existing["source_sha256"] != hashes:
        parser.error("Sweep sources changed; choose a new output")
    out.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(cfg, indent=2))
    snapshot_sources(out, hashes)
    meta_path.write_text(json.dumps({**metadata(), "source_sha256": hashes,
        "started_utc": existing.get("started_utc", datetime.now(timezone.utc).isoformat()),
        "requested_methods": ["q_learning"]}, indent=2))
    rows = []
    for alpha in (.05, .2, .5):
        for decay in (.99, .995, .999):
            run_cfg = copy.deepcopy(cfg)
            run_cfg["q_learning"]["alpha"] = alpha
            run_cfg["training"]["epsilon_decay"] = decay
            records = []
            for seed in cfg["training"]["seeds"]:
                folder = out / f"a_{alpha}_decay_{decay}" / f"seed_{seed}"
                if args.resume and (folder / "result.json").exists():
                    if json.loads((folder / "config.json").read_text()) != run_cfg:
                        parser.error(f"Per-run configuration mismatch: {folder}")
                    record = json.loads((folder / "result.json").read_text())
                else:
                    record = train_one(run_cfg, "q_learning", seed, folder, resume=args.resume)
                records.append(record)
            stable = stats([r["stable_optimal_check"] for r in records if r["stable_optimal_check"] is not None])
            steps = stats([r["environment_steps"] for r in records])
            rows.append({"alpha": alpha, "epsilon_decay": decay,
                         "decays_to_min": epsilon_decay_steps(cfg["training"]["epsilon_start"], cfg["training"]["epsilon_min"], decay),
                         "optimal_policies": sum(r["optimal"] for r in records), "n_runs": len(records),
                         "stable_n": stable["n"], "stable_mean": stable["mean"], "stable_std": stable["std"],
                         "environment_steps_mean": steps["mean"], "environment_steps_std": steps["std"]})
    write_csv(out / "sweep_summary.csv", rows)


if __name__ == "__main__":
    main()

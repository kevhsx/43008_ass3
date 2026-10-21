"""Run the fixed configuration, keeping every result linked to its inputs."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from warehouse.experiment import train_one, random_baseline, metadata
from warehouse.env import WarehouseEnv
from warehouse.configuration import validate_config
from warehouse.provenance import source_hashes, snapshot_sources
import gymnasium as gym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="results/main")
    parser.add_argument("--methods", nargs="+", choices=["q_learning", "dqn", "random"], default=["q_learning", "dqn", "random"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs and restore episode-boundary checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.episodes is not None:
        cfg["training"]["episodes"] = args.episodes
    if args.seeds is not None:
        cfg["training"]["seeds"] = args.seeds
    try:
        validate_config(cfg)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.checkpoint_interval < 1:
        parser.error("checkpoint interval must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        parser.error("Output contains a different config; choose a new directory")
    config_path.write_text(json.dumps(cfg, indent=2))
    from gymnasium.utils.env_checker import check_env
    for mode in ("discrete", "vector"):
        for render in ("ansi", "rgb_array"):
            checked_env = gym.make("WarehouseNav-v0", settings=cfg["environment"], observation_mode=mode, render_mode=render)
            check_env(checked_env.unwrapped)
            checked_env.close()
    hashes = source_hashes()
    meta_path = out / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("source_sha256") != hashes:
        parser.error("Source differs from recorded experiment; use a new output directory")
    snapshot_sources(out, hashes)
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    methods = sorted(set(existing.get("requested_methods", [])) | set(args.methods))
    meta_path.write_text(json.dumps({**metadata(), "started_utc": existing.get("started_utc", datetime.now(timezone.utc).isoformat()),
                                   "source_sha256": hashes, "requested_methods": methods,
                                   "checkpoint_interval": args.checkpoint_interval,
                                   "check_env": "passed (registered environment, both observations and render modes)"}, indent=2))
    for method in args.methods:
        if method == "random":
            if not (args.resume and (out / "random_episodes.csv").exists()):
                random_baseline(cfg, out)
            continue
        for seed in cfg["training"]["seeds"]:
            folder = out / method / f"seed_{seed}"
            if args.resume and (folder / "result.json").exists():
                if json.loads((folder / "config.json").read_text()) != cfg:
                    raise ValueError("Per-run configuration mismatch")
                continue
            train_one(cfg, method, seed, folder, resume=args.resume, checkpoint_interval=args.checkpoint_interval)
    from warehouse.analysis import analyse
    analyse(out)


if __name__ == "__main__":
    main()

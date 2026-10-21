"""Save a copy of the code used for each experiment."""
import hashlib
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def guard_output_directory(out, resume=False, config_name="config.json"):
    """Check the output folder before writing results."""
    out = Path(out)
    if out.exists() and not out.is_dir():
        raise ValueError(f"Output is not a directory: {out}")
    if resume:
        if not (out / config_name).is_file() or not (out / "metadata.json").is_file():
            raise ValueError(f"Resume requires an existing run with {config_name} and metadata.json")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Output is not empty; choose a new directory or use --resume for the same run")


def source_hashes():
    paths = [*ROOT.glob("*.py"), *ROOT.joinpath("warehouse").glob("*.py"), *ROOT.joinpath("tests").glob("*.py")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def snapshot_sources(out, hashes):
    for name, expected in hashes.items():
        source = ROOT / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed during experiment setup: {name}")
        target = Path(out) / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot conflict: {name}")
        if not target.exists():
            shutil.copyfile(source, target)


def training_hashes():
    names = ["warehouse/agents.py", "warehouse/env.py", "warehouse/experiment.py",
             "warehouse/checkpoint.py", "warehouse/configuration.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def recorded_run_folders(out, method, seeds):
    """Require exactly the configured seeds before reporting a completed experiment."""
    parent = Path(out) / method
    expected = {f"seed_{seed}" for seed in seeds}
    actual = {p.name for p in parent.glob("seed_*") if p.is_dir()}
    missing, extra = expected - actual, actual - expected
    if missing:
        raise ValueError(f"Missing expected run: {parent / sorted(missing)[0]}")
    if extra:
        raise ValueError(f"Unexpected run: {parent / sorted(extra)[0]}")
    folders = [parent / f"seed_{seed}" for seed in seeds]
    model = "model.pt" if method == "dqn" else "q_table.npy"
    for folder in folders:
        for name in ("config.json", "result.json", "training.csv", "greedy_checks.csv", model):
            if not (folder / name).is_file():
                raise ValueError(f"Run is incomplete: missing {folder / name}")
    return folders

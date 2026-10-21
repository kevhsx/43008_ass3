"""Archive the code used for a run without rewriting historical experiment records."""
import hashlib
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


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

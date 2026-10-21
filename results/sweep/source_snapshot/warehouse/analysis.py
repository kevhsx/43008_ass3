"""Report descriptive statistics with explicit units and censored convergence."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from .env import WarehouseEnv
from .experiment import write_csv


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def stats(values):
    a = np.array(values, dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"n": len(a), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else None,
            "min": float(a.min()), "max": float(a.max())}


def fmt(s, digits=2):
    if not s["n"]:
        return "not observed"
    mean = f"{s['mean']:.{digits}f}"
    return mean + (f" ± {s['std']:.{digits}f}" if s["std"] is not None else " (n=1)")


def rolling(a, window=50):
    # Partial windows at the beginning; no padded observations.
    sums = np.r_[0.0, np.cumsum(a)]
    ends = np.arange(1, len(a) + 1)
    starts = np.maximum(0, ends - window)
    return (sums[ends] - sums[starts]) / (ends - starts)


def analyse(out):
    out = Path(out)
    cfg = json.loads((out / "config.json").read_text())
    env = WarehouseEnv(cfg["environment"])
    summary = {"optimal_steps": env.optimal_steps, "optimal_return": env.optimal_return,
               "trained": {}, "random": None, "training_windows": {}}
    flat, methods = [], {}
    for method in ("q_learning", "dqn"):
        folders = sorted((out / method).glob("seed_*"))
        records = [json.loads((p / "result.json").read_text()) for p in folders if (p / "result.json").exists()]
        if not records:
            continue
        methods[method] = records
        s = {"n_policies": len(records), "successful_policies": sum(r["success"] for r in records),
             "optimal_policies": sum(r["optimal"] for r in records), "sampling_unit": "independently trained final policy"}
        for key in ("return", "steps", "collision_rate", "environment_steps", "training_seconds", "evaluation_seconds",
                    "first_success_check", "stable_optimal_check", "steps_to_stable_optimal"):
            s[key] = stats([r[key] for r in records if r[key] is not None])
        s["collision_rate_pooled"] = sum(r["collisions"] for r in records) / sum(r["steps"] for r in records)
        s["successful_steps"] = stats([r["steps"] for r in records if r["success"]])
        summary["trained"][method] = s
        for r in records:
            flat.append({k: v for k, v in r.items() if k != "route"})
        summary["training_windows"][method] = []
        for lo, hi in ((1, 100), (101, 500), (2501, 3000)):
            if hi > cfg["training"]["episodes"]:
                continue
            values = {k: [] for k in ("return", "success", "steps", "collision_rate")}
            for folder in folders:
                if not (folder / "result.json").exists():
                    continue
                rows = [r for r in read_csv(folder / "training.csv") if lo <= int(r["episode"]) <= hi]
                for key in ("return", "success", "steps"):
                    values[key].append(np.mean([float(r[key]) for r in rows]))
                values["collision_rate"].append(sum(int(r["collisions"]) for r in rows) / sum(int(r["steps"]) for r in rows))
            summary["training_windows"][method].append({"episodes": [lo, hi], **{k: stats(v) for k, v in values.items()}})
    if (out / "random_episodes.csv").exists():
        rows = read_csv(out / "random_episodes.csv")
        seeds = sorted(set(r["seed"] for r in rows))
        successes = [r for r in rows if int(r["success"])]
        summary["random"] = {"n_episodes": len(rows), "successes": len(successes),
                             "episode_success_rate": len(successes) / len(rows),
                             "sampling_unit": "episode (seed groups reported separately)",
                             "seed_success_rate": stats([np.mean([int(r["success"]) for r in rows if r["seed"] == seed]) for seed in seeds]),
                             "successful_steps": stats([float(r["steps"]) for r in successes]),
                             "collision_rate_pooled": sum(int(r["collisions"]) for r in rows) / sum(int(r["steps"]) for r in rows),
                             **{key: stats([float(r[key]) for r in rows]) for key in ("return", "steps", "collision_rate")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if flat:
        write_csv(out / "final_policies.csv", flat)
    lines = ["# Reproduced experimental results", "", f"BFS optimum: {env.optimal_steps} steps; undiscounted return {env.optimal_return:.2f}.",
             "", "Final checkpoint only; epsilon=0; one rollout per independently trained policy.", "",
             "| Method | Successful policies | Optimal policies | Return | Steps | Pooled collisions | Training seconds |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, s in summary["trained"].items():
        n = s["n_policies"]
        lines.append(f"| {method} | {s['successful_policies']}/{n} | {s['optimal_policies']}/{n} | {fmt(s['return'])} | {fmt(s['steps'])} | {s['collision_rate_pooled']:.2%} | {fmt(s['training_seconds'], 3)} |")
    if summary["random"]:
        r = summary["random"]
        lines += ["", f"Random baseline: {r['successes']}/{r['n_episodes']} episodes ({r['episode_success_rate']:.1%}); return {fmt(r['return'])}; "
                  f"all-episode steps {fmt(r['steps'], 1)}; successful steps {fmt(r['successful_steps'], 1)} (n={r['successful_steps']['n']}); "
                  f"pooled collision rate {r['collision_rate_pooled']:.2%}."]
    lines += ["", "| Method | First successful check | Stable-optimal check | Runs reaching stable optimum | Steps to stable optimum | Total train steps |",
              "|---|---:|---:|---:|---:|---:|"]
    for method, s in summary["trained"].items():
        lines.append(f"| {method} | {fmt(s['first_success_check'], 1)} | {fmt(s['stable_optimal_check'], 1)} | {s['stable_optimal_check']['n']}/{s['n_policies']} | {fmt(s['steps_to_stable_optimal'], 0)} | {fmt(s['environment_steps'], 0)} |")
    lines += ["", "Standard deviations use ddof=1. Convergence means include only runs where the event was observed; failures are censored, not assigned zero. Stable means optimal at every subsequent recorded check through the training horizon. Checks occur every 10 episodes plus the final episode. Collision rates pool blocked moves over attempted moves.",
              "", "## Training windows", "", "Values below are mean ± sample std of per-seed window statistics. Exploration is included.", "",
              "| Method | Episodes | Return | Success | Steps | Collision rate |", "|---|---|---:|---:|---:|---:|"]
    for method, windows in summary["training_windows"].items():
        for w in windows:
            lines.append(f"| {method} | {w['episodes'][0]}–{w['episodes'][1]} | {fmt(w['return'])} | {fmt(w['success'], 3)} | {fmt(w['steps'], 1)} | {fmt(w['collision_rate'], 4)} |")
    lines += ["", "Timing: training loop, including in-memory logging; excludes greedy checks, model creation, final evaluation and disk writes. See metadata.json for hardware, software versions and source hashes.",
              "", "The rollout cap is treated as an external sampling cutoff of the underlying goal-reaching task. Both algorithms bootstrap on truncation; delivery termination alone masks the target. The measured success metric is delivery within 200 moves."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    if methods:
        plot_learning(out, methods)
        plot_routes(out, methods, env)


def plot_learning(out, methods):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method in methods:
        label = {"q_learning": "Q-Learning", "dqn": "DQN"}[method]
        folders = sorted((out / method).glob("seed_*"))
        logs = [read_csv(p / "training.csv") for p in folders if (p / "result.json").exists()]
        checks = [read_csv(p / "greedy_checks.csv") for p in folders if (p / "result.json").exists()]
        for ax, key in zip(axes[0], ("return", "success")):
            ys = np.array([rolling([float(r[key]) for r in log]) for log in logs])
            x = np.arange(1, ys.shape[1] + 1)
            line, = ax.plot(x, ys.mean(0), label=label)
            if len(ys) > 1:
                sd = ys.std(0, ddof=1)
                lower, upper = ys.mean(0)-sd, ys.mean(0)+sd
                if key == "success":
                    lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
                ax.fill_between(x, lower, upper, color=line.get_color(), alpha=.15)
        x = [int(r["episode"]) for r in checks[0]]
        steps = np.array([[int(r["steps"]) for r in c] for c in checks])
        line, = axes[1, 0].plot(x, np.median(steps, 0), label=label)
        axes[1, 0].fill_between(x, steps.min(0), steps.max(0), color=line.get_color(), alpha=.15)
        success = np.array([[int(r["success"]) for r in c] for c in checks])
        axes[1, 1].plot(x, success.mean(0), label=label)
    optimum = json.loads((out / "summary.json").read_text())["optimal_steps"]
    axes[1, 0].axhline(optimum, ls="--", color="black", label=f"BFS optimum ({optimum})")
    for ax, title in zip(axes.flat, ("Training return (50-episode trailing mean ± std)",
                                   "Training success (mean ± std, band clipped to [0,1])",
                                   "Greedy steps (median and min–max)", "Greedy delivery success across training runs")):
        ax.set_title(title)
        ax.set_xlabel("Training episode")
        ax.grid(alpha=.2)
        ax.legend()
    axes[0, 0].set_ylabel("Undiscounted return")
    axes[0, 1].set_ylabel("Success fraction")
    axes[0, 1].set_ylim(-.025, 1.025)
    axes[1, 0].set_ylabel("Attempted moves (failures include cap)")
    axes[1, 1].set_ylabel("Fraction of independently trained policies")
    fig.savefig(out / "learning_curves.png", dpi=180)
    plt.close(fig)


def plot_routes(out, methods, env):
    fig, axes = plt.subplots(1, len(methods), figsize=(7*len(methods), 6), squeeze=False, constrained_layout=True)
    for ax, (method, records) in zip(axes[0], methods.items()):
        grid = np.zeros((env.height, env.width))
        for x, y in env.shelves:
            grid[y, x] = 1
        ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, alpha=.6)
        for i, r in enumerate(records):
            route = np.array(r["route"], dtype=float)
            route += (i-(len(records)-1)/2)*.025
            ax.plot(route[:, 0], route[:, 1], alpha=.75, label=f"seed {r['seed']}: {r['steps']} moves, {'delivered' if r['success'] else 'failed'}")
        for p, text in ((env.start, "S"), (env.pickup, "P"), (env.delivery, "D")):
            ax.text(*p, text, weight="bold", fontsize=13, bbox={"facecolor":"white", "alpha":.8, "edgecolor":"none"})
        ax.set_xticks(range(env.width)); ax.set_yticks(range(env.height))
        ax.set_xticks(np.arange(-.5, env.width), minor=True)
        ax.set_yticks(np.arange(-.5, env.height), minor=True)
        ax.grid(which="minor", alpha=.3)
        label = {"q_learning": "Q-Learning", "dqn": "DQN"}[method]
        ax.set_title(f"{label}: final greedy policies")
        ax.set_xlabel("Column x"); ax.set_ylabel("Row y")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), fontsize=8, ncol=2)
    fig.savefig(out / "final_routes.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    analyse(Path(sys.argv[1]))

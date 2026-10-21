# Warehouse Robot Navigation

GitHub-ready project with source code, tests, recorded experiments, final models, and an executed analysis notebook. See [upload instructions](docs/GITHUB_UPLOAD.md) and [experiment records](docs/EXPERIMENT_RECORDS.md).

Team 100 compares Q-Learning, DQN and a random policy on a pickup-and-delivery task. The code was built from the supplied project reports. It uses a local Gymnasium environment and Pygame; ROS and physical robots are not included.


## 目录导航与重要程度

先读本页运行说明；需要理解代码时，按 `env.py → agents.py → experiment.py` 的顺序阅读。详细职责、依赖和修改位置见 [架构说明](docs/architecture.md)。

| 类别 | 文件或目录 | 保留建议 |
|---|---|---|
| 核心实现 | `warehouse/` | 必须保留；各模块协同工作 |
| 运行入口 | `demo.py`、`run_experiments.py`、`verify_results.py` | 保留，用于演示、训练和验证 |
| 实验入口 | `run_sweep.py`、`run_adaptation.py`、`reproduce.py` | 完整项目保留；仅播放模型时不需要 |
| 配置与安装 | `config.json`、`requirements*.txt` | 保留；notebook 依赖仅重跑笔记本时需要 |
| 质量保证 | `tests/`、`pytest.ini` | 提交仓库保留；仅运行演示时不需要 |
| 实验依据 | `results/main/`、`results/sweep/`、`results/adaptation/` | 报告与复现保留；含日志、最终模型和历史源码 |
| 分析与图表 | `output/`、`tools/` | 完整交付保留；仅播放模型时不需要 |
| 文档 | `docs/` | 使用与维护参考，运行程序不依赖 |
| 可选便利工具 | `scripts/macos/` | 仅 macOS 双击安装或播放时使用 |
| Git 配置 | `.gitignore`、`.gitattributes`、`.git/` | Git 工作流程使用；`.git/` 不通过网页上传 |

“仅播放时不需要”不等于可以从课程项目中删除。实验目录里的 `source_snapshot/` 是历史证据，不是重复的当前代码；请保持原样。

## Setup

Use Python 3.12, the version tested for this project. Create an environment on each computer; do not share .venv or copy a link to another computer's environment.

```bash
python3 -m venv .venv
```

macOS / Linux:

```bash
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python demo.py
```

Windows:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe demo.py
```

In VS Code, select the Python interpreter inside this project's .venv. The Run button must use that interpreter. On macOS, scripts/macos/setup_environment.command installs the packages and scripts/macos/start_demo.command starts playback. If a copied .venv is a link to another project, create a separate environment before sharing the project.

## Playback

```bash
.venv/bin/python demo.py --method q_learning --seed 0
.venv/bin/python demo.py --method dqn --seed 0
.venv/bin/python demo.py --method manual
.venv/bin/python demo.py --results results/adaptation --method dqn --seed 0
```

Space pauses, R resets and Esc quits. Manual mode uses the arrow keys. Playback reads saved models; it does not train them. Relative demo paths are based on the script folder.

## Training and checks

Run these commands from the project folder:

```bash
.venv/bin/python run_experiments.py
.venv/bin/python run_experiments.py --output results/my_run
.venv/bin/python run_experiments.py --output results/my_run --resume
.venv/bin/python run_sweep.py --output results/new_sweep
.venv/bin/python run_sweep.py --output results/new_sweep --resume
.venv/bin/python run_adaptation.py --main-results results/main --output results/new_adaptation --episodes 1000
.venv/bin/python -m pytest -q
.venv/bin/python verify_results.py --results results/main --sweep results/sweep
.venv/bin/python verify_results.py --results results/adaptation --adaptation results/adaptation --base-results results/main
.venv/bin/python reproduce.py --output results/reproduction --include-adaptation
```

Without --output, main training uses a new timestamped folder in results/runs. Main, sweep and adaptation entry points reject nonempty output folders; use --resume only for the same run. Resume requires matching configuration and source files. Checkpoints are saved every 100 episodes by default; unsaved episodes run again after recovery. They contain model, optimizer, replay, exploration and random states.

## Files

| File or folder | Purpose |
|---|---|
| config.json | Map, rewards, seeds and training settings |
| warehouse/env.py | Movement, rewards and BFS reference route |
| warehouse/agents.py | Q-Learning and DQN |
| warehouse/experiment.py | Training loop and evaluation |
| warehouse/analysis.py | Statistics and plots |
| warehouse/checkpoint.py | Training recovery |
| warehouse/configuration.py | Input checks |
| warehouse/provenance.py | Source snapshots and output protection |
| tests/ | Automated checks |
| results/main | Original main experiment |
| results/sweep | Nine Q-Learning settings, ten seeds each |
| results/adaptation | Changed-map experiment |
| output/jupyter-notebook | Executed analysis notebook |
| output/figures | Report figures |
| docs/EXPERIMENT_RECORDS.md | Which records to keep |

## Results and limits

On the main map, Q-Learning succeeds for 10/10 seeds with optimal 18-step routes. DQN succeeds for 8/10 seeds, with 7/10 optimal routes. The random baseline succeeds in 68/1000 episodes. Failed seeds are included in the report.

The changed map adds shelves at (3,5) and (4,6). Frozen policies fail for all ten seeds of both methods. After 1000 extra episodes, both methods reach 10/10 optimal 20-step routes. DQN keeps its weights but resets replay and Adam. This is one route-blocking test; it does not show zero-shot generalisation or an advantage over training from scratch.

The main task is a fixed 10x10 map with four actions and position/carrying observations. Each step costs 0.1, a collision adds a penalty of 1, pickup gives 5 and delivery gives 20. The 200-step cap truncates a rollout; only delivery ends bootstrapping. BFS is an evaluation reference, not input to either learner. Final comparison uses the last model, not the best intermediate checkpoint.

Main settings were fixed before training. The Q sweep describes sensitivity; no completed DQN pilot search is claimed. Training time excludes evaluation and disk writes and depends on hardware.

Historical source snapshots match the recorded experiments. Current code includes later checks and recovery features; these changes do not replace the recorded results. Source changes may prevent resuming an older incomplete run. Use a fresh output folder when testing modified code.

## Record validation

Formal analysis requires exactly the configured seeds and complete training logs. Requested random baselines must include every configured seed/episode exactly once. Missing, empty or duplicated records fail validation instead of producing a partial report. Greedy checks must match the configured evaluation schedule, including the final episode, and agree with training-log seed and cumulative interaction counts. Report-figure export validates these inputs before writing figures. An empty decays_to_min field in a new sweep means the exploration floor is only approached, not reached in a finite number of mathematical decay steps; zero means the schedule starts at its floor.

## Analysis notebook

The notebook has seven executed code cells. To execute it again, install requirements-notebook.txt and select the project's Python kernel. Saved outputs can be read without rerunning it.

## References

Environment interface: https://gymnasium.farama.org/introduction/create_custom_env/

DQN background: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

## Repository contents and new experiments

The published `results/main`, `results/sweep`, and `results/adaptation` directories retain final models, raw logs, configurations, and historical source snapshots. Recovery checkpoints are omitted: saved final models support playback and verification, but cannot resume historical training with optimizer and replay state. Create a new virtual environment after cloning.

Treat published experiment folders as read-only evidence. Main training creates a fresh folder by default. For sweeps and adaptation, supply a new output folder. Both entry points reject nonempty folders unless --resume is explicitly supplied. New sweeps retain source snapshots and metadata; historical sweeps without metadata can be verified for policy behaviour but do not provide verified source hashes. Never train into the published directories.

The original snapshots and metadata are preserved, including original machine paths. They describe historical experiments; use the explicit `--base-results results/main` override when verifying relocated adaptation records. Current code includes the record-safety and completeness fixes described in [development notes](docs/development.md). Historical snapshots identify the implementation used for the published experiments.

This repository does not include assignment briefs, student reports, local review notes, credentials, or a Python virtual environment. No redistribution license is assigned by this packaging step.

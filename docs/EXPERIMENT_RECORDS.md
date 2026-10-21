# Experiment Records

Keep the evidence behind the Part C tables and figures until marking and the presentation are finished. Back up important runs separately; this project does not upload files automatically.

| Record | Keep it? | Reason |
|---|---|---|
| training.csv, greedy_checks.csv, random_episodes.csv | Yes | Raw observations used for statistics |
| config.json, metadata.json, source_snapshot | Yes | Settings, software versions and code used for the run |
| q_table.npy, model.pt | Yes | Final policies for playback and verification |
| result.json, summary.json, verification.json | Yes | Routes, summaries and checks |
| checkpoint.zip | While training; optional afterwards | Full training state for recovery |
| Report, figures and analysis notebook | Yes | Submission and explanation of results |
| Old screenshots and unused drafts | Optional | Presentation history |
| __pycache__, .pytest_cache | No | Automatically recreated |
| .venv | Keep locally; do not submit | Install a new environment on each computer |

The report uses results/main, results/sweep and results/adaptation. Keep these directories together with their source snapshots. New main runs default to separate timestamped folders in results/runs. Do not mix statistics from different experiments.

Use --resume with the original --output to continue an interrupted run. Keep the code and settings unchanged during training. Final models alone cannot restore optimizer, replay and random states.

For team work, use Git for source, tests, configuration and text instructions. Store large model/checkpoint collections in a shared drive or Git LFS. After an experiment finishes, verify it and copy the complete run folder to the shared location. Avoid two computers writing to the same experiment folder.

## Packaged repository

This copy includes the three recorded experiments and their small final policy files. Recovery checkpoints, local environments and caches are excluded. Internal report review notes are omitted. Keep historical snapshots unchanged. See GITHUB_UPLOAD.md before publishing or running new experiments.

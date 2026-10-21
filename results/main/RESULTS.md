# Reproduced experimental results

BFS optimum: 18 steps; undiscounted return 23.20.

Final checkpoint only; epsilon=0; one rollout per independently trained policy.

| Method | Successful policies | Optimal policies | Return | Steps | Pooled collisions | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| q_learning | 10/10 | 10/10 | 23.20 ± 0.00 | 18.00 ± 0.00 | 0.00% | 0.314 ± 0.004 |
| dqn | 8/10 | 7/10 | 14.50 ± 18.18 | 55.00 ± 76.44 | 0.00% | 42.364 ± 20.522 |

Random baseline: 68/1000 episodes (6.8%); return -58.74 ± 14.72; all-episode steps 195.7 ± 19.1; successful steps 136.7 ± 40.7 (n=68); pooled collision rate 22.14%.

| Method | First successful check | Stable-optimal check | Runs reaching stable optimum | Steps to stable optimum | Total train steps |
|---|---:|---:|---:|---:|---:|
| q_learning | 53.0 ± 8.2 | 53.0 ± 8.2 | 10/10 | 9102 ± 1056 | 71346 ± 603 |
| dqn | 1416.2 ± 781.4 | 1681.4 ± 642.6 | 7/10 | 266932 ± 151649 | 370376 ± 174393 |

Standard deviations use ddof=1. Convergence means include only runs where the event was observed; failures are censored, not assigned zero. Stable means optimal at every subsequent recorded check through the training horizon. Checks occur every 10 episodes plus the final episode. Collision rates pool blocked moves over attempted moves.

## Training windows

Values below are mean ± sample std of per-seed window statistics. Exploration is included.

| Method | Episodes | Return | Success | Steps | Collision rate |
|---|---|---:|---:|---:|---:|
| q_learning | 1–100 | -18.86 ± 1.68 | 0.629 ± 0.032 | 128.7 ± 4.0 | 0.1800 ± 0.0041 |
| q_learning | 101–500 | 20.76 ± 0.29 | 1.000 ± 0.000 | 27.1 ± 0.6 | 0.0568 ± 0.0083 |
| q_learning | 2501–3000 | 22.94 ± 0.04 | 1.000 ± 0.000 | 19.0 ± 0.1 | 0.0084 ± 0.0017 |
| dqn | 1–100 | -51.57 ± 2.08 | 0.036 ± 0.036 | 198.2 ± 2.2 | 0.1710 ± 0.0110 |
| dqn | 101–500 | -26.57 ± 9.72 | 0.076 ± 0.222 | 188.1 ± 35.4 | 0.0539 ± 0.0142 |
| dqn | 2501–3000 | 13.52 ± 18.83 | 0.799 ± 0.421 | 56.4 ± 75.7 | 0.0203 ± 0.0178 |

Timing: training loop, including in-memory logging; excludes greedy checks, model creation, final evaluation and disk writes. See metadata.json for hardware, software versions and source hashes.

The rollout cap is treated as an external sampling cutoff of the underlying goal-reaching task. Both algorithms bootstrap on truncation; delivery termination alone masks the target. The measured success metric is delivery within 200 moves.

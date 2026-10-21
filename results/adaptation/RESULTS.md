# Reproduced experimental results

BFS optimum: 20 steps; undiscounted return 23.00.

Final checkpoint only; epsilon=0; one rollout per independently trained policy.

| Method | Successful policies | Optimal policies | Return | Steps | Pooled collisions | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| q_learning | 10/10 | 10/10 | 23.00 ± 0.00 | 20.00 ± 0.00 | 0.00% | 0.173 ± 0.005 |
| dqn | 10/10 | 10/10 | 23.00 ± 0.00 | 20.00 ± 0.00 | 0.00% | 5.989 ± 1.318 |

| Method | First successful check | Stable-optimal check | Runs reaching stable optimum | Steps to stable optimum | Total train steps |
|---|---:|---:|---:|---:|---:|
| q_learning | 68.0 ± 14.0 | 69.0 ± 16.6 | 10/10 | 11961 ± 1957 | 37109 ± 822 |
| dqn | 180.0 ± 90.6 | 302.0 ± 169.8 | 10/10 | 37207 ± 15748 | 54141 ± 11843 |

Standard deviations use ddof=1. Convergence means include only runs where the event was observed; failures are censored, not assigned zero. Stable means optimal at every subsequent recorded check through the training horizon. Checks occur every 10 episodes plus the final episode. Collision rates pool blocked moves over attempted moves.

## Training windows

Values below are mean ± sample std of per-seed window statistics. Exploration is included.

| Method | Episodes | Return | Success | Steps | Collision rate |
|---|---|---:|---:|---:|---:|
| q_learning | 1–100 | -28.36 ± 2.37 | 0.515 ± 0.049 | 144.3 ± 6.6 | 0.1889 ± 0.0069 |
| q_learning | 101–500 | 19.91 ± 0.32 | 1.000 ± 0.001 | 30.1 ± 0.8 | 0.0685 ± 0.0071 |
| dqn | 1–100 | -45.98 ± 3.94 | 0.195 ± 0.117 | 185.7 ± 10.0 | 0.1775 ± 0.0171 |
| dqn | 101–500 | 9.81 ± 8.23 | 0.844 ± 0.153 | 61.5 ± 27.0 | 0.0880 ± 0.0131 |

Timing: training loop, including in-memory logging; excludes greedy checks, model creation, final evaluation and disk writes. See metadata.json for hardware, software versions and source hashes.

The rollout cap is treated as an external sampling cutoff of the underlying goal-reaching task. Both algorithms bootstrap on truncation; delivery termination alone masks the target. The measured success metric is delivery within 200 moves.

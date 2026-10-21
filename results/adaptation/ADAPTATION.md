# Changed-layout transfer and adaptation

Two additional shelves at (3,5) and (4,6) block the original approaches to pickup. BFS optimum: 20 moves; undiscounted return 23.0.

This is a deliberately route-blocking stress test, not a representative random sample of warehouses. The new layout was excluded from the original main experiment and was not used to retune either algorithm.

Each of the ten base final checkpoints per method was evaluated frozen, then warm-started for 1000 additional episodes. DQN weights are retained; replay and Adam start fresh. Exploration restarts at 1.0 with the original decay.

| Method | Frozen successful policies | Adapted successful policies | Adapted optimal policies | Adapted return | Additional environment steps | Additional training seconds |
|---|---:|---:|---:|---:|---:|---:|
| q_learning | 0/10 | 10/10 | 10/10 | 23.00 ± 0.00 | 37109 ± 822 | 0.17 ± 0.00 |
| dqn | 0/10 | 10/10 | 10/10 | 23.00 ± 0.00 | 54141 ± 11843 | 5.99 ± 1.32 |

Adapted-policy metrics do not replace the main-layout metrics. A successful adaptation demonstrates relearning after new interaction, not zero-shot generalisation or faster learning than training from scratch. No from-scratch comparator or diverse map suite is included.

"""Validate experimental inputs before creating output files or starting runs."""
import math


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def validate_environment(env):
    for name in ("width", "height"):
        positive_int(env[name], name, 2)
    positive_int(env["max_steps"], "max_steps")
    points = [env[name] for name in ("start", "pickup", "delivery")] + list(env["shelves"])
    for point in points:
        if len(point) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in point):
            raise ValueError("Coordinates must be pairs of integers")
    if len(set(map(tuple, env["shelves"]))) != len(env["shelves"]):
        raise ValueError("Shelf cells must be unique")
    for name in ("step_reward", "collision_reward", "pickup_reward", "delivery_reward"):
        finite(env[name], name)
    if env["step_reward"] >= 0 or env["collision_reward"] > 0:
        raise ValueError("Movement must have negative reward and collisions cannot earn a bonus")
    if env["pickup_reward"] <= 0 or env["delivery_reward"] <= 0:
        raise ValueError("Pickup and delivery rewards must be positive")


def validate_config(cfg):
    validate_environment(cfg["environment"])
    training = cfg["training"]
    for name in ("episodes", "evaluation_interval", "random_episodes_per_seed"):
        positive_int(training[name], name)
    seeds = training["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    for seed in seeds:
        positive_int(seed, "seed", 0)
    for name in ("epsilon_start", "epsilon_min", "epsilon_decay"):
        finite(training[name], name)
    if not 0 <= training["epsilon_min"] <= training["epsilon_start"] <= 1:
        raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1")
    if not 0 < training["epsilon_decay"] <= 1:
        raise ValueError("epsilon_decay must be in (0,1]")
    q, dqn = cfg["q_learning"], cfg["dqn"]
    for method in (q, dqn):
        finite(method["gamma"], "gamma")
        if not 0 <= method["gamma"] < 1:
            raise ValueError("gamma must be in [0,1)")
    finite(q["alpha"], "alpha")
    if not 0 < q["alpha"] <= 1:
        raise ValueError("alpha must be in (0,1]")
    if not dqn["hidden_sizes"]:
        raise ValueError("DQN needs at least one hidden layer")
    for size in dqn["hidden_sizes"]:
        positive_int(size, "hidden size")
    for name in ("batch_size", "buffer_size", "learning_starts", "train_frequency"):
        positive_int(dqn[name], name)
    if dqn["batch_size"] > dqn["buffer_size"]:
        raise ValueError("Replay buffer must accommodate a full minibatch")
    for name in ("learning_rate", "target_tau", "gradient_clip"):
        finite(dqn[name], name)
        if dqn[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if dqn["target_tau"] > 1:
        raise ValueError("target_tau must be <= 1")

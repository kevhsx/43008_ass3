"""Gymnasium environment; coordinates are (column, row), with row zero at top."""
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 6}
    MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def __init__(self, settings, observation_mode="discrete", render_mode=None):
        super().__init__()
        self.settings = dict(settings)
        self.width, self.height = settings["width"], settings["height"]
        self.start, self.pickup, self.delivery = (
            tuple(settings[k]) for k in ("start", "pickup", "delivery")
        )
        self.shelves = frozenset(map(tuple, settings["shelves"]))
        self.max_steps = settings["max_steps"]
        if self.width < 2 or self.height < 2 or self.max_steps < 1:
            raise ValueError("Dimensions must be >= 2; max_steps must be positive")
        for point in (*self.shelves, self.start, self.pickup, self.delivery):
            if not (0 <= point[0] < self.width and 0 <= point[1] < self.height):
                raise ValueError("Coordinates outside grid")
        if len({self.start, self.pickup, self.delivery}) != 3:
            raise ValueError("Start, pickup and delivery must be distinct")
        if any(p in self.shelves for p in (self.start, self.pickup, self.delivery)):
            raise ValueError("Task locations must be free cells")
        self.optimal_route = self.shortest_path(self.start, self.pickup) + self.shortest_path(self.pickup, self.delivery)[1:]
        self.optimal_steps = len(self.optimal_route) - 1
        self.optimal_return = (settings["pickup_reward"] + settings["delivery_reward"]
                               + self.optimal_steps * settings["step_reward"])
        if self.optimal_steps > self.max_steps:
            raise ValueError("Step cap cannot accommodate a valid delivery")
        if observation_mode not in ("discrete", "vector"):
            raise ValueError("observation_mode must be discrete or vector")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("Unsupported render mode")
        self.observation_mode, self.render_mode = observation_mode, render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = (spaces.Discrete(self.width * self.height * 2)
                                  if observation_mode == "discrete" else
                                  spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32))
        self.reset()

    def shortest_path(self, start, goal):
        queue, previous = deque([start]), {start: None}
        while queue:
            p = queue.popleft()
            if p == goal:
                path = []
                while p is not None:
                    path.append(p)
                    p = previous[p]
                return path[::-1]
            for dx, dy in self.MOVES:
                n = (p[0] + dx, p[1] + dy)
                if (0 <= n[0] < self.width and 0 <= n[1] < self.height
                        and n not in self.shelves and n not in previous):
                    previous[n] = p
                    queue.append(n)
        raise ValueError("Task locations are not connected")

    def observation(self):
        if self.observation_mode == "discrete":
            return int(self.x + self.width * self.y + self.width * self.height * self.carrying)
        return np.array([self.x / (self.width - 1), self.y / (self.height - 1),
                         float(self.carrying)], dtype=np.float32)

    def info(self, collision=False, success=False):
        return {"position": (self.x, self.y), "carrying": self.carrying,
                "steps": self.steps, "collisions": self.collisions,
                "collision": collision, "success": success}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y = self.start
        self.carrying, self.steps, self.collisions, self.finished = False, 0, 0, False
        return self.observation(), self.info()

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset after termination or truncation")
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 3")
        dx, dy = self.MOVES[int(action)]
        nx, ny = self.x + dx, self.y + dy
        collision = not (0 <= nx < self.width and 0 <= ny < self.height) or (nx, ny) in self.shelves
        self.steps += 1
        reward = float(self.settings["step_reward"])
        if collision:
            self.collisions += 1
            reward += self.settings["collision_reward"]
        else:
            self.x, self.y = nx, ny
        if (self.x, self.y) == self.pickup and not self.carrying:
            self.carrying = True
            reward += self.settings["pickup_reward"]
        success = (self.x, self.y) == self.delivery and self.carrying
        if success:
            reward += self.settings["delivery_reward"]
        terminated = bool(success)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self.finished = terminated or truncated
        return self.observation(), float(reward), terminated, truncated, self.info(collision, success)

    def render(self):
        if self.render_mode == "rgb_array":
            scale = 40
            grid = np.full((self.height, self.width, 3), 245, dtype=np.uint8)
            for x, y in self.shelves:
                grid[y, x] = (75, 85, 95)
            for p, c in [(self.start, (150, 180, 220)), (self.pickup, (250, 200, 80)),
                         (self.delivery, (90, 190, 120)), ((self.x, self.y), (60, 100, 220))]:
                grid[p[1], p[0]] = c
            return grid.repeat(scale, 0).repeat(scale, 1)
        if self.render_mode == "ansi":
            grid = [["." for _ in range(self.width)] for _ in range(self.height)]
            for x, y in self.shelves:
                grid[y][x] = "#"
            for p, symbol in [(self.start, "S"), (self.pickup, "P"), (self.delivery, "D"),
                              ((self.x, self.y), "R" if self.carrying else "r")]:
                grid[p[1]][p[0]] = symbol
            return "\n".join(" ".join(row) for row in grid)

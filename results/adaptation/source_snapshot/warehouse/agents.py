"""Q-Learning and plain DQN; terminal transitions alone suppress bootstrapping."""
import numpy as np


class QLearning:
    def __init__(self, states, settings):
        self.q = np.zeros((states, 4), dtype=np.float64)
        self.alpha, self.gamma = settings["alpha"], settings["gamma"]

    def greedy(self, state):
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state, terminated):
        target = reward if terminated else reward + self.gamma * self.q[next_state].max()
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def save(self, path):
        np.save(path, self.q)


class DQN:
    def __init__(self, settings, seed):
        import torch
        from torch import nn
        self.torch, self.settings = torch, dict(settings)
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 731]))
        layers, inputs = [], 3
        for hidden in settings["hidden_sizes"]:
            layers.extend([nn.Linear(inputs, hidden), nn.ReLU()])
            inputs = hidden
        layers.append(nn.Linear(inputs, 4))
        self.online = nn.Sequential(*layers)
        import copy
        self.target = copy.deepcopy(self.online).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=settings["learning_rate"])
        self.loss_fn = nn.SmoothL1Loss()
        size = settings["buffer_size"]
        self.states = np.zeros((size, 3), dtype=np.float32)
        self.next_states = np.zeros((size, 3), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.terminated = np.zeros(size, dtype=np.float32)
        self.count, self.updates = 0, 0

    def greedy(self, state):
        with self.torch.no_grad():
            return int(self.online(self.torch.as_tensor(state)).argmax().item())

    def update(self, state, action, reward, next_state, terminated):
        t, cfg = self.torch, self.settings
        index = self.count % cfg["buffer_size"]
        self.states[index], self.actions[index] = state, action
        self.rewards[index], self.next_states[index], self.terminated[index] = reward, next_state, terminated
        self.count += 1
        if (self.count < max(cfg["learning_starts"], cfg["batch_size"])
                or self.count % cfg["train_frequency"]):
            return None
        ids = self.rng.choice(min(self.count, cfg["buffer_size"]), cfg["batch_size"], replace=False)
        states, next_states = t.from_numpy(self.states[ids]), t.from_numpy(self.next_states[ids])
        actions = t.from_numpy(self.actions[ids]).unsqueeze(1)
        rewards, terminals = t.from_numpy(self.rewards[ids]), t.from_numpy(self.terminated[ids])
        predictions = self.online(states).gather(1, actions).squeeze(1)
        with t.no_grad():
            targets = rewards + cfg["gamma"] * (1 - terminals) * self.target(next_states).max(1).values
        loss = self.loss_fn(predictions, targets)
        if not t.isfinite(loss):
            raise FloatingPointError("Non-finite DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.online.parameters(), cfg["gradient_clip"])
        self.optimizer.step()
        with t.no_grad():
            for target, online in zip(self.target.parameters(), self.online.parameters()):
                target.lerp_(online, cfg["target_tau"])
        self.updates += 1
        return float(loss.item())

    def save(self, path):
        self.torch.save({"model": self.online.state_dict(), "settings": self.settings}, path)

    def load(self, path):
        saved = self.torch.load(path, map_location="cpu", weights_only=True)
        self.online.load_state_dict(saved["model"])
        self.target.load_state_dict(saved["model"])

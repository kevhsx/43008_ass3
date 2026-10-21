"""Save and restore training state between episodes."""
import io
import json
import os
from pathlib import Path
import zipfile
import numpy as np


def save_checkpoint(path, agent, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("state.json", json.dumps(state))
        buffer = io.BytesIO()
        if state["method"] == "q_learning":
            np.save(buffer, agent.q, allow_pickle=False)
            archive.writestr("q.npy", buffer.getvalue())
        else:
            t = agent.torch
            payload = {"online": agent.online.state_dict(), "target": agent.target.state_dict(),
                       "optimizer": agent.optimizer.state_dict(), "count": agent.count,
                       "updates": agent.updates, "replay_rng": agent.rng.bit_generator.state,
                       "torch_rng": t.get_rng_state(),
                       "replay": {name: t.from_numpy(getattr(agent, name)) for name in
                                  ("states", "next_states", "actions", "rewards", "terminated")}}
            t.save(payload, buffer)
            archive.writestr("dqn.pt", buffer.getvalue())
    os.replace(temporary, path)


def load_checkpoint(path, agent, cfg, method, seed, source_hashes):
    with zipfile.ZipFile(path) as archive:
        state = json.loads(archive.read("state.json"))
        if (state["config"] != cfg or state["method"] != method or state["seed"] != seed
                or state["training_source_sha256"] != source_hashes):
            raise ValueError("Checkpoint configuration, seed, method or training code mismatch")
        if method == "q_learning":
            agent.q = np.load(io.BytesIO(archive.read("q.npy")), allow_pickle=False)
        else:
            t = agent.torch
            payload = t.load(io.BytesIO(archive.read("dqn.pt")), map_location="cpu", weights_only=True)
            agent.online.load_state_dict(payload["online"])
            agent.target.load_state_dict(payload["target"])
            agent.optimizer.load_state_dict(payload["optimizer"])
            agent.count, agent.updates = payload["count"], payload["updates"]
            agent.rng.bit_generator.state = payload["replay_rng"]
            t.set_rng_state(payload["torch_rng"])
            for name, array in payload["replay"].items():
                setattr(agent, name, array.numpy().copy())
    return state

"""Warehouse pickup-and-delivery reinforcement learning project."""
from gymnasium.envs.registration import register, registry

if "WarehouseNav-v0" not in registry:
    register(id="WarehouseNav-v0", entry_point="warehouse.env:WarehouseEnv")

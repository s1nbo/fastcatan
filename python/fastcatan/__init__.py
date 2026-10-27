"""High-throughput Catan simulator (C++ core via nanobind)."""

from importlib.metadata import version as _version

from ._fastcatan import (  # noqa: F401
    BatchedEnv,
    Env,
    MASK_WORDS,
    NUM_ACTIONS,
    NUM_PLAYERS,
    NUM_RESOURCES,
    NUM_NODES,
    NUM_EDGES,
    NUM_HEXES,
    NUM_PORTS,
    NO_PLAYER,
    NO_ACTION,
    SNAPSHOT_BYTES,
    MAX_ACTION_OUTCOMES,
    STATE_VIEW_VERSION,
    action,
)

__all__ = [
    "BatchedEnv",
    "Env",
    "MASK_WORDS",
    "NUM_ACTIONS",
    "NUM_PLAYERS",
    "NUM_RESOURCES",
    "NUM_NODES",
    "NUM_EDGES",
    "NUM_HEXES",
    "NUM_PORTS",
    "NO_PLAYER",
    "NO_ACTION",
    "SNAPSHOT_BYTES",
    "MAX_ACTION_OUTCOMES",
    "STATE_VIEW_VERSION",
    "action",
]

__version__ = _version("fastcatan")

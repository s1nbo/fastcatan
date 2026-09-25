"""The Python boundary rejects invalid indices and untrusted snapshots."""
from __future__ import annotations

import pytest
import numpy as np

import fastcatan


def test_env_requires_reset() -> None:
    env = fastcatan.Env()
    with pytest.raises(RuntimeError, match="not reset"):
        _ = env.phase
    with pytest.raises(RuntimeError, match="not reset"):
        env.step(fastcatan.action.ROLL_DICE)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda env: env.player_vp(fastcatan.NUM_PLAYERS), "seat"),
        (lambda env: env.player_resource(0, fastcatan.NUM_RESOURCES), "resource"),
        (lambda env: env.player_dev(0, 5), "card"),
        (lambda env: env.node(fastcatan.NUM_NODES), "node"),
        (lambda env: env.edge(fastcatan.NUM_EDGES), "edge"),
        (lambda env: env.hex_number(fastcatan.NUM_HEXES), "hex"),
        (lambda env: env.port_type(fastcatan.NUM_PORTS), "port"),
    ],
)
def test_env_indices_are_checked(call, message: str) -> None:
    env = fastcatan.Env()
    env.reset(0)
    with pytest.raises(RuntimeError, match=message):
        call(env)


def test_corrupt_snapshot_is_rejected_without_mutating_env() -> None:
    env = fastcatan.Env()
    env.reset(9)
    before = env.snapshot()

    with pytest.raises(RuntimeError, match="invalid snapshot"):
        env.load_snapshot(bytes([0xFF]) * fastcatan.SNAPSHOT_BYTES)

    assert env.snapshot() == before


def test_legal_actions_matches_mask() -> None:
    env = fastcatan.Env()
    env.reset(4)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    env.action_mask(mask)
    from_mask = [
        action
        for action in range(fastcatan.NUM_ACTIONS)
        if (int(mask[action >> 6]) >> (action & 63)) & 1
    ]
    assert env.legal_actions() == from_mask

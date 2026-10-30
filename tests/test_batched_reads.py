"""Batched actor and status reads agree with the per-env getters."""
from __future__ import annotations

import random

import numpy as np
import pytest

import fastcatan
from tests.conftest import legal_actions


N = 32
PLAYERS = fastcatan.NUM_PLAYERS
PHASE_ENDED = 3
STATUS = {name: offset for name, (offset, _) in fastcatan.STATUS_FIELDS.items()}


def test_status_layout_constants() -> None:
    assert fastcatan.STATUS_INTS == 4
    assert sorted(STATUS.values()) == list(range(fastcatan.STATUS_INTS))
    assert set(STATUS) == {"done", "winner", "turn_count", "current_player"}


def test_buffer_shapes_are_checked() -> None:
    batch = fastcatan.BatchedEnv(2)
    with pytest.raises(RuntimeError, match="not reset"):
        batch.write_actors(np.zeros(2, dtype=np.uint8))
    batch.reset()
    with pytest.raises(RuntimeError, match="actor buffer"):
        batch.write_actors(np.zeros(1, dtype=np.uint8))
    with pytest.raises(RuntimeError, match="status buffer"):
        batch.write_status(np.zeros((1, fastcatan.STATUS_INTS), dtype=np.int32))
    with pytest.raises(RuntimeError, match="status buffer"):
        batch.write_status(np.zeros((2, 3), dtype=np.int32))


def test_actors_and_status_match_per_env_getters() -> None:
    batch = fastcatan.BatchedEnv(N, seed=13)
    batch.reset()

    masks = np.zeros((N, fastcatan.MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(N, dtype=np.uint32)
    dones = np.zeros(N, dtype=np.uint8)
    actors = np.zeros(N, dtype=np.uint8)
    status = np.zeros((N, fastcatan.STATUS_INTS), dtype=np.int32)
    rng = random.Random(13)

    # A random game runs a few thousand actions, so this budget gives every
    # slot time to finish at least one game and be autoreset.
    seen_done = 0
    for _ in range(12_000):
        batch.write_actors(actors)
        batch.write_status(status)
        for i in range(N):
            assert actors[i] == batch.actor_to_act(i)
            row = status[i]
            assert row[STATUS["done"]] == int(batch.phase(i) == PHASE_ENDED)
            assert row[STATUS["winner"]] == (
                batch.last_winner(i) if batch.phase(i) == PHASE_ENDED
                else fastcatan.NO_PLAYER)
            assert row[STATUS["turn_count"]] == batch.turn_count(i)
            assert row[STATUS["current_player"]] == batch.current_player(i)

        batch.write_masks(masks)
        for i in range(N):
            actions[i] = rng.choice(legal_actions(masks[i]))
        batch.step_autoreset(actions, dones)
        seen_done += int(dones.sum())
        if seen_done >= N:
            break
    assert seen_done >= N, "no games finished, terminal status never checked"


def test_status_winner_matches_the_finished_game() -> None:
    batch = fastcatan.BatchedEnv(N, seed=31)
    batch.reset()
    masks = np.zeros((N, fastcatan.MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(N, dtype=np.uint32)
    dones = np.zeros(N, dtype=np.uint8)
    status = np.zeros((N, fastcatan.STATUS_INTS), dtype=np.int32)
    rng = random.Random(31)

    finished = 0
    single = fastcatan.Env()
    for _ in range(40_000):
        batch.write_masks(masks)
        for i in range(N):
            if batch.phase(i) == PHASE_ENDED:
                actions[i] = fastcatan.NO_ACTION
                continue
            actions[i] = rng.choice(legal_actions(masks[i]))
        batch.step(actions, dones)
        if dones.all():
            break

    batch.write_status(status)
    for i in range(N):
        row = status[i]
        assert row[STATUS["done"]] == 1
        single.load_snapshot(batch.snapshot(i))
        assert row[STATUS["winner"]] == single.winner
        assert row[STATUS["turn_count"]] == single.turn_count
        assert row[STATUS["current_player"]] == single.current_player
        winner = int(row[STATUS["winner"]])
        if winner != fastcatan.NO_PLAYER:
            assert single.player_vp(winner) >= 10
            finished += 1
    assert finished > 0, "no game produced a winner"

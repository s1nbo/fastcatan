"""Batched simulator lifecycle and parity checks."""
from __future__ import annotations

import numpy as np
import pytest

import fastcatan
from tests.conftest import play_random_game


N = 8


def _first_legal(mask: np.ndarray) -> int:
    for word_idx, word in enumerate(mask):
        bits = int(word)
        if bits:
            return word_idx * 64 + (bits & -bits).bit_length() - 1
    raise AssertionError("live game has no legal move")


def test_batched_reset_step_and_snapshot() -> None:
    batch = fastcatan.BatchedEnv(N, seed=7)
    batch.reset()

    masks = np.zeros((N, fastcatan.MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(N, dtype=np.uint32)
    dones = np.zeros(N, dtype=np.uint8)
    batch.write_masks(masks)
    for i in range(N):
        actions[i] = _first_legal(masks[i])

    batch.step(actions, dones)
    assert not dones.any()

    env = fastcatan.Env()
    for i in range(N):
        env.load_snapshot(batch.snapshot(i))
        assert env.phase == batch.phase(i)
        assert env.current_player == batch.current_player(i)
        assert env.actor_to_act == batch.actor_to_act(i)
        assert env.state_view(0) == batch.state_view(i, 0)
        for player in range(fastcatan.NUM_PLAYERS):
            assert env.player_vp(player) == batch.player_vp(i, player)
            assert env.player_handsize(player) == batch.player_handsize(i, player)


def test_batched_requires_reset_and_valid_indices() -> None:
    with pytest.raises(RuntimeError, match="num_envs"):
        fastcatan.BatchedEnv(0)

    batch = fastcatan.BatchedEnv(2)
    with pytest.raises(RuntimeError, match="not reset"):
        batch.phase(0)

    batch.reset()
    with pytest.raises(RuntimeError, match="env_idx"):
        batch.phase(2)
    with pytest.raises(RuntimeError, match="player"):
        batch.player_vp(0, fastcatan.NUM_PLAYERS)


def test_batched_buffer_shapes_are_checked() -> None:
    batch = fastcatan.BatchedEnv(2)
    batch.reset()

    with pytest.raises(RuntimeError, match="mask buffer"):
        batch.write_masks(np.zeros((1, fastcatan.MASK_WORDS), dtype=np.uint64))
    with pytest.raises(RuntimeError, match="step buffer"):
        batch.step(np.zeros(1, dtype=np.uint32), np.zeros(2, dtype=np.uint8))
    with pytest.raises(RuntimeError, match="snapshot buffer"):
        batch.save_snapshots(
            np.zeros((1, fastcatan.SNAPSHOT_BYTES), dtype=np.uint8)
        )
    with pytest.raises(RuntimeError, match="snapshot buffer"):
        batch.load_snapshots(
            np.zeros((1, fastcatan.SNAPSHOT_BYTES), dtype=np.uint8)
        )
    with pytest.raises(RuntimeError, match="seeds"):
        batch.reseed(np.zeros(1, dtype=np.uint64))

    actions = np.zeros(2, dtype=np.uint32)
    actions.flags.writeable = False
    batch.step(actions, np.zeros(2, dtype=np.uint8))


def test_bulk_snapshot_restore_and_reseed_match_single_env() -> None:
    source = fastcatan.Env()
    source.reset(5)
    while source.phase != 2:
        source.step(source.legal_actions()[0])
    assert source.dice_roll == 0

    snapshots = np.empty((N, fastcatan.SNAPSHOT_BYTES), dtype=np.uint8)
    snapshots[:] = np.frombuffer(source.snapshot(), dtype=np.uint8)
    snapshots.flags.writeable = False

    batch = fastcatan.BatchedEnv(N, seed=99)
    batch.load_snapshots(snapshots)
    seeds = np.arange(100, 100 + N, dtype=np.uint64)
    seeds.flags.writeable = False
    batch.reseed(seeds)

    actions = np.full(N, fastcatan.action.ROLL_DICE, dtype=np.uint32)
    dones = np.zeros(N, dtype=np.uint8)
    batch.step(actions, dones)
    assert not dones.any()

    saved = np.zeros((N, fastcatan.SNAPSHOT_BYTES), dtype=np.uint8)
    batch.save_snapshots(saved)
    single = fastcatan.Env()
    for i in range(N):
        single.load_snapshot(source.snapshot())
        single.reseed(int(seeds[i]))
        single.step(fastcatan.action.ROLL_DICE)
        assert saved[i].tobytes() == single.snapshot()
        assert saved[i].tobytes() == batch.snapshot(i)


def test_invalid_bulk_snapshot_is_atomic() -> None:
    batch = fastcatan.BatchedEnv(2)
    batch.reset()
    before = [batch.snapshot(i) for i in range(2)]
    corrupt = np.full(
        (2, fastcatan.SNAPSHOT_BYTES), 0xFF, dtype=np.uint8
    )

    with pytest.raises(RuntimeError, match="invalid snapshot"):
        batch.load_snapshots(corrupt)

    assert [batch.snapshot(i) for i in range(2)] == before


def test_normal_step_preserves_terminal_and_autoreset_is_explicit() -> None:
    terminal, _ = play_random_game(0)
    winner = terminal.winner
    data = np.frombuffer(terminal.snapshot(), dtype=np.uint8)
    snapshots = np.tile(data, (2, 1))

    batch = fastcatan.BatchedEnv(2, seed=123)
    batch.load_snapshots(snapshots)
    actions = np.full(2, fastcatan.NO_ACTION, dtype=np.uint32)
    dones = np.zeros(2, dtype=np.uint8)

    before = [batch.snapshot(i) for i in range(2)]
    batch.step(actions, dones)
    assert dones.tolist() == [1, 1]
    assert [batch.snapshot(i) for i in range(2)] == before
    assert [batch.last_winner(i) for i in range(2)] == [winner, winner]

    batch.step_autoreset(actions, dones)
    assert dones.tolist() == [1, 1]
    assert [batch.phase(i) for i in range(2)] == [0, 0]
    assert [batch.last_winner(i) for i in range(2)] == [winner, winner]


def test_no_action_preserves_live_slot() -> None:
    batch = fastcatan.BatchedEnv(1)
    batch.reset()
    before = batch.snapshot(0)
    actions = np.array([fastcatan.NO_ACTION], dtype=np.uint32)
    dones = np.ones(1, dtype=np.uint8)
    batch.step(actions, dones)
    assert dones.tolist() == [0]
    assert batch.snapshot(0) == before


def test_native_random_rollout_completes_exact_games_deterministically() -> None:
    first = fastcatan.BatchedEnv(N, seed=19)
    second = fastcatan.BatchedEnv(N, seed=19)
    first.reset()
    second.reset()

    first_result = first.run_random_games(32)
    second_result = second.run_random_games(32)

    assert first_result == second_result
    assert first_result[0] == 32
    assert first_result[1] > first_result[0]
    assert [first.snapshot(i) for i in range(N)] == [
        second.snapshot(i) for i in range(N)
    ]


def test_native_random_rollout_validates_lifecycle_and_count() -> None:
    batch = fastcatan.BatchedEnv(2)
    with pytest.raises(RuntimeError, match="not reset"):
        batch.run_random_games(1)

    batch.reset()
    with pytest.raises(RuntimeError, match="positive"):
        batch.run_random_games(0)

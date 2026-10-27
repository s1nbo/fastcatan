"""Smoke: simulator constructs, resets, and random games terminate with a winner."""
from __future__ import annotations

import pytest

import fastcatan
from tests.conftest import play_random_game


def test_constants_sane():
    assert fastcatan.NUM_PLAYERS == 4
    assert fastcatan.NUM_NODES == 54
    assert fastcatan.NUM_EDGES == 72
    assert fastcatan.NUM_HEXES == 19
    assert fastcatan.NUM_PORTS == 9
    assert fastcatan.MASK_WORDS * 64 >= fastcatan.NUM_ACTIONS


def test_env_construct_and_reset():
    env = fastcatan.Env()
    env.reset(0)
    assert env.phase in (0, 1, 2, 3)
    assert env.current_player in range(4)
    assert env.turn_count == 0


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123, 2024])
def test_random_game_terminates(seed):
    env, steps = play_random_game(seed)
    assert env.phase == 3, "expected ENDED phase after termination"
    assert env.winner == env.current_player, "only the turn owner can win"
    assert env.player_vp(env.winner) >= 10
    assert steps > 0

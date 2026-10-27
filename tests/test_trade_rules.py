"""Player-to-player trade rule regressions."""

from __future__ import annotations

import random

import fastcatan
from tests.conftest import find_state


def _state_with_give_action() -> tuple[fastcatan.Env, int]:
    rng = random.Random(17)
    env = fastcatan.Env()
    env.reset(17)

    for _ in range(2_000):
        legal = env.legal_actions()
        give = [
            action for action in legal
            if fastcatan.action.TRADE_ADD_GIVE_BASE
            <= action < fastcatan.action.TRADE_ADD_GIVE_BASE + 5
        ]
        if give:
            return env, give[0] - fastcatan.action.TRADE_ADD_GIVE_BASE
        assert legal
        done = env.step(rng.choice(legal))
        assert not done

    raise AssertionError("seed-17 fixture never reached trade composition")


def test_trade_cannot_request_a_resource_already_offered() -> None:
    env, resource = _state_with_give_action()
    env.step(fastcatan.action.TRADE_ADD_GIVE_BASE + resource)

    legal = env.legal_actions()
    same_resource_want = fastcatan.action.TRADE_ADD_WANT_BASE + resource
    assert same_resource_want not in legal

    before = env.snapshot()
    done = env.step(same_resource_want)
    assert env.snapshot() == before
    assert int(done) == 0


def test_trade_keeps_turn_owner_while_each_opponent_answers() -> None:
    a = fastcatan.action

    def next_seat_can_accept(env, legal):
        responder = (env.current_player + 1) % 4
        return any(
            give != want and env.player_resource(responder, want) > 0
            for give in range(5) if a.TRADE_ADD_GIVE_BASE + give in legal
            for want in range(5)
        )

    env = find_state(next_seat_can_accept)
    proposer = env.current_player
    order = [(proposer + i) % 4 for i in (1, 2, 3)]
    give, want = next(
        (give, want)
        for give in range(5) if a.TRADE_ADD_GIVE_BASE + give in env.legal_actions()
        for want in range(5)
        if give != want and env.player_resource(order[0], want) > 0
    )
    hands = {seat: [env.player_resource(seat, r) for r in range(5)]
             for seat in (proposer, order[0])}

    env.step(a.TRADE_ADD_GIVE_BASE + give)
    env.step(a.TRADE_ADD_WANT_BASE + want)
    env.step(a.TRADE_OPEN)

    for index, seat in enumerate(order):
        assert env.current_player == proposer, "the turn owner must not change during a trade"
        assert env.actor_to_act == seat, "opponents answer clockwise from the proposer"
        assert sorted(env.legal_actions()) == [a.TRADE_ACCEPT, a.TRADE_DECLINE]
        assert env.state_view(seat)["legal_actions"] == env.legal_actions()
        assert env.state_view(proposer)["current_player"] == proposer
        env.step(a.TRADE_ACCEPT if index == 0 else a.TRADE_DECLINE)

    assert env.actor_to_act == proposer
    assert sorted(env.legal_actions()) == [a.TRADE_CONFIRM_BASE + order[0], a.TRADE_CANCEL]
    env.step(a.TRADE_CONFIRM_BASE + order[0])

    assert env.flag == 0
    assert env.current_player == env.actor_to_act == proposer
    assert a.END_TURN in env.legal_actions(), "the proposer keeps the rest of their turn"
    for seat, sign in ((proposer, 1), (order[0], -1)):
        expected = list(hands[seat])
        expected[give] -= sign
        expected[want] += sign
        assert [env.player_resource(seat, r) for r in range(5)] == expected

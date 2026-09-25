"""Player-visible semantic state never exposes an opponent's private cards."""
from __future__ import annotations

import random

import pytest

import fastcatan


def _buy_dev_state(seed: int = 123) -> fastcatan.Env:
    env = fastcatan.Env()
    env.reset(seed)
    rng = random.Random(seed)
    for _ in range(5_000):
        legal = env.legal_actions()
        if fastcatan.action.BUY_DEV in legal:
            return env
        assert not env.step(rng.choice(legal))
    raise AssertionError("no development-card purchase became legal")


def test_state_view_has_only_requested_players_private_state() -> None:
    env = fastcatan.Env()
    env.reset(7)

    for pov in range(fastcatan.NUM_PLAYERS):
        view = env.state_view(pov)
        assert view["schema_version"] == fastcatan.STATE_VIEW_VERSION
        assert view["pov"] == pov
        assert view["discarding_player"] == env.discarding_player
        assert view["actor_to_act"] == env.actor_to_act
        assert len(view["players"]) == fastcatan.NUM_PLAYERS
        for player in view["players"]:
            private_fields = (
                player["resources"],
                player["development_cards"],
                player["development_cards_bought_this_turn"],
            )
            if player["seat"] == pov:
                assert all(value is not None for value in private_fields)
            else:
                assert private_fields == (None, None, None)

        if pov == env.actor_to_act:
            assert view["legal_actions"] == env.legal_actions()
        else:
            assert view["legal_actions"] is None

    with pytest.raises(RuntimeError, match="pov"):
        env.state_view(fastcatan.NUM_PLAYERS)


def test_hidden_development_draw_identity_does_not_change_opponent_view() -> None:
    env = _buy_dev_state()
    buyer = env.actor_to_act
    observer = (buyer + 1) % fastcatan.NUM_PLAYERS
    outcomes = env.enumerate_outcomes(fastcatan.action.BUY_DEV)
    assert len(outcomes) >= 2

    child = fastcatan.Env()
    opponent_views = []
    buyer_views = []
    for _, snapshot in outcomes:
        child.load_snapshot(snapshot)
        opponent_views.append(child.state_view(observer))
        buyer_views.append(child.state_view(buyer))

    assert all(view == opponent_views[0] for view in opponent_views[1:])
    assert any(view != buyer_views[0] for view in buyer_views[1:])


def test_unopened_trade_draft_is_visible_only_to_owner() -> None:
    env = _buy_dev_state(17)
    owner = env.actor_to_act
    observer = (owner + 1) % fastcatan.NUM_PLAYERS

    give_action = next(
        action for action in env.legal_actions()
        if fastcatan.action.TRADE_ADD_GIVE_BASE
        <= action
        < fastcatan.action.TRADE_ADD_GIVE_BASE + fastcatan.NUM_RESOURCES
    )
    give_resource = give_action - fastcatan.action.TRADE_ADD_GIVE_BASE
    want_resource = (give_resource + 1) % fastcatan.NUM_RESOURCES
    want_action = fastcatan.action.TRADE_ADD_WANT_BASE + want_resource

    env.step(give_action)
    env.step(want_action)
    assert env.state_view(observer)["trade"]["give"] is None
    assert env.state_view(owner)["trade"]["give"][give_resource] == 1

    env.step(fastcatan.action.TRADE_OPEN)
    public_trade = env.state_view(observer)["trade"]
    assert public_trade["give"][give_resource] == 1
    assert public_trade["want"][want_resource] == 1

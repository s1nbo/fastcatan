# Changelog

All notable changes to FastCatan are documented here.

## 2.1.0 - 2026-10-29

Native observation encoding and event logging for reinforcement-learning
loops over large batches. No rule changes: outcomes are identical to 2.0.1 for
the same seed and actions.

### Added

- `write_obs` on `Env` and `BatchedEnv` encodes a game as `OBS_SIZE` float32
  features from one seat's point of view, with that seat at relative slot 0.
  `write_obs_pov` takes a seat per game. The encoding exposes public
  information plus the observing seat's own private information only, and is
  versioned by `OBS_VERSION`.
- `BatchedEnv.write_actors` and `BatchedEnv.write_status` read the acting seat
  and a `done`/`winner`/`turn_count`/`current_player` row for every game
  without a per-game call.
- An opt-in native event log: `enable_log`, `disable_log`, `drain_events`,
  `pending_events`, `log_overflow`, `log_mask`, and `log_capacity_per_env`.
  `LOG_TRADES` records player trades, `LOG_GAMES` records game boundaries, and
  `LOG_ACTIONS` records every `(actor, action)`, which replays a game exactly
  when combined with the recorded seed. Records are fixed-width int32 rows
  described by `EVENT_FIELDS` and versioned by `EVENT_VERSION`.
- `BatchedEnv` per-game episode bookkeeping: `game_id`, `game_seed`, and
  `step_in_game`.
- `BatchedEnv` getters mirroring `Env` for parity checks and debugging: `flag`,
  `turn_count`, `player_vp_public`, `trade_proposer`, `trade_compose_count`,
  `trade_give`, `trade_want`, and `trade_response`.
- Field-layout maps `OBS_FIELDS`, `EVENT_FIELDS`, and `STATUS_FIELDS`.

### Changed

- `run_random_games` now raises while event recording is enabled. It plays
  whole games without entering the batched step loop, so it cannot record
  events, and failing is better than returning an empty log.

### Notes

The observation encoder is a port of the 1084-float encoder from the
pre-2.0 line, corrected for two information leaks: the per-type development
deck remainder is replaced by the deck size, since the breakdown reveals which
card types opponents hold, and an unopened trade draft is now hidden from every
seat but its owner. With the actor flag, the fourth trade response, and the
compose budget added, `OBS_SIZE` is 1089.

The event log is not part of a snapshot. Enable it before `reset` to capture
the first episode's start.

## 2.0.1 - 2026-10-27

Rule fixes that follow the [catan.com base-game FAQ](https://www.catan.com/faq/basegame).
Game outcomes can differ from 2.0.0 for the same seed and actions.

### Fixed

- A player now wins only on their own turn. Anyone who reaches 10 VP during
  another player's turn wins when their own turn begins. `winner` reports the
  turn owner instead of the lowest seat with 10 VP.
- Longest Road: when the holder loses the card and several players tie for the
  longest road, nobody receives it until one player is longest alone. A holder
  who is only tied keeps the card.
- Longest Road no longer depends on build order. A road counts up to an
  opponent's settlement or city but never continues through it, and new roads
  cannot be built through one.
- When the bank cannot cover production of a resource owed to a single player,
  that player receives what the bank has left.
- `player_road_length` is tracked from the first setup road.
- `current_player` stays the turn owner while opponents answer a player trade;
  `actor_to_act` names the seat that must respond.

Snapshots of an open player trade from 2.0.0 are rejected.

## 2.0.0 - 2026-09-25

This is a simulator-only release and contains intentional breaking API changes.

### Added

- Exact physical outcome enumeration for dice rolls, development-card draws,
  and robber steals.
- Player-safe semantic state views with private information hidden by seat.
- Bulk batch snapshot save/restore, reseeding, and an explicit `NO_ACTION`.
- A terminal-preserving batch step and an explicit `step_autoreset` operation.
- Snapshot validation at the Python boundary.

### Changed

- `Env.step(action)` now returns only whether the game is terminal.
- Terminal batched states remain inspectable until explicitly reset.
- Package metadata and examples now describe only the simulator API.

### Removed

- Tensor-specific state encoders.
- Configurable scoring and shaped-return machinery.
- Search diagnostics and decision-layer examples.

Snapshots remain version-local and should not be moved between 1.x and 2.x.

## 1.0.2 - 2026-08-11

- Last 1.x release published to PyPI.

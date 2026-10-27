# Changelog

All notable changes to FastCatan are documented here.

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

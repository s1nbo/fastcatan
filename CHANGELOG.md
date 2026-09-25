# Changelog

All notable changes to FastCatan are documented here.

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

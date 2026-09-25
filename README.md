# fastcatan

[![PyPI](https://img.shields.io/pypi/v/fastcatan.svg)](https://pypi.org/project/fastcatan/)
[![Python](https://img.shields.io/pypi/pyversions/fastcatan.svg)](https://pypi.org/project/fastcatan/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deterministic Settlers of Catan simulator with a C++ core and Python bindings
through [nanobind](https://github.com/wjakob/nanobind).

- Complete four-player game state and rule transitions.
- Seeded, reproducible board setup and chance events.
- Legal-action validation with strict no-op behavior for illegal actions.
- Snapshot and restore support for replay and debugging.
- Optional batched stepping with OpenMP.

## Install

```bash
pip install fastcatan
```

Prebuilt wheels target Linux x86_64 and macOS Apple Silicon on CPython 3.12+.
Other supported systems build from the source distribution and need a C++23
compiler and CMake 3.27 or newer.

## Simulate a game

```python
import random
import fastcatan

rng = random.Random(0)
game = fastcatan.Env()
game.reset(seed=0)

while not game.done:
    action = rng.choice(game.legal_actions())
    game.step(action)

print("winner:", game.winner)
print("victory points:", [
    game.player_vp(seat) for seat in range(fastcatan.NUM_PLAYERS)
])
```

`step(action)` returns whether the game is terminal. An action that is not in
`legal_actions()` leaves the game unchanged. `action_mask(out)` exposes the same
legal set as a packed NumPy bitset when allocation-free access is needed.

The semantic state API includes board tiles, nodes, edges, bank contents,
development cards, player inventories, trade state, the robber, awards, and the
acting seat. `snapshot()` and `load_snapshot()` preserve the complete game and
random-number state.

## Batched simulation

```python
import numpy as np
import fastcatan

n = 1024
games = fastcatan.BatchedEnv(n, seed=0)
games.reset()

masks = np.zeros((n, fastcatan.MASK_WORDS), dtype=np.uint64)
actions = np.zeros(n, dtype=np.uint32)
dones = np.zeros(n, dtype=np.uint8)

games.write_masks(masks)
# Fill actions with one legal action per game.
games.step(actions, dones)
```

`step` preserves terminal states so they can be inspected or saved. Use
`step_autoreset` for continuous simulation; it records the completed game's
winning seat in `last_winner(index)` before resetting that slot. `NO_ACTION`
leaves an independently progressing slot unchanged.

`save_snapshots`, `load_snapshots`, and `reseed` operate on whole batches without
per-game Python calls. They use arrays shaped `(n, SNAPSHOT_BYTES)` for snapshots
and `(n,)` for seeds.

## Exact chance outcomes

The simulator can enumerate physical chance transitions without selecting what
any player should do:

```python
for probability, snapshot in game.enumerate_outcomes(action):
    successor = fastcatan.Env()
    successor.load_snapshot(snapshot)
```

Dice rolls, development-card draws, and robber steals produce weighted
successors. Deterministic actions produce one successor, and probabilities sum
to one. Each successor also advances its stored random-number state exactly as
a sampled transition would. This keeps Catan's probability rules in the
simulator while leaving any decision-making logic to downstream packages.

## Player-visible state

`state_view(seat)` returns unnormalized semantic data for a player. It includes
public board and player information plus that seat's own resources and
development cards. Opponents' private fields are `None`, unopened trade drafts
remain private, and legal actions are included only for the acting seat. The
dictionary schema is versioned by `STATE_VIEW_VERSION`.

## Main API

| Symbol | Purpose |
|---|---|
| `Env` | One game: reset, inspect, list legal actions, step, snapshot, restore |
| `BatchedEnv` | Contiguous collection of independently progressing games |
| `action` | Action-ID constants |
| `NUM_ACTIONS`, `MASK_WORDS` | Legal-action bitset dimensions |
| `NUM_PLAYERS`, `NUM_RESOURCES` | Game-wide counts |
| `NUM_NODES`, `NUM_EDGES`, `NUM_HEXES`, `NUM_PORTS` | Board dimensions |
| `NO_PLAYER` | Sentinel used when no seat owns an award or terminal result |
| `NO_ACTION` | Explicit no-op sentinel for a batched slot |
| `SNAPSHOT_BYTES` | Snapshot size for the current package version |
| `MAX_ACTION_OUTCOMES` | Maximum successors from exact chance enumeration |
| `STATE_VIEW_VERSION` | Semantic player-view schema version |

Snapshots are an internal binary format. Only load snapshots created by the
same package version; malformed snapshots are rejected.

## Upgrading from 1.x

Version 2.0 is a simulator-only release with a smaller public API:

- `Env.step(action)` returns only the terminal-state boolean.
- `BatchedEnv.step` preserves terminal states. Use `step_autoreset` when a
  completed slot should immediately start a new game.
- Tensor encoders, configurable scoring, and search diagnostics are no longer
  part of this package.
- `state_view`, `enumerate_outcomes`, and bulk snapshot methods provide neutral
  simulator data for downstream projects.

See [CHANGELOG.md](CHANGELOG.md) for the complete release summary.

## Development

```bash
pip install .
pytest
```

The standalone C++ invariant harness can also be built and run:

```bash
cmake -S . -B build -DFASTCATAN_ARCH=off
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the uniformly random-agent benchmark to measure completed games per second:

```bash
python examples/benchmark_simulator.py
```

The random policy and complete-game rollout run entirely in C++, so the result
measures simulator throughput rather than Python action-selection overhead. Use
`--envs`, `--games`, and `--seed` to change the batch size, completed-game count,
and random seed.

## License

[MIT](LICENSE) © 2026 s1nbo

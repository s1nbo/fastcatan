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
- Batched observation, actor, and status reads that need no per-game calls.
- An opt-in native event log for trades, actions, and game boundaries.

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

## Batched reads

Every batched read fills a caller-provided NumPy array in place, releases the
GIL, allocates nothing, and parallelizes with OpenMP. A training loop over
thousands of games therefore needs no per-game Python call at all.

| Call | Buffer | Contents |
|---|---|---|
| `write_masks(out)` | `uint64[n, MASK_WORDS]` | legal-action bitset |
| `write_actors(out)` | `uint8[n]` | `actor_to_act` per game |
| `write_status(out)` | `int32[n, STATUS_INTS]` | see `STATUS_FIELDS` |
| `write_obs(out)` | `float32[n, OBS_SIZE]` | observation from each game's own actor |
| `write_obs_pov(seats, out)` | `float32[n, OBS_SIZE]` | observation from `seats[i]` |

`STATUS_FIELDS` maps a column name to `(offset, length)`:

| Column | Offset | Value |
|---|---|---|
| `done` | 0 | 1 once the game is terminal |
| `winner` | 1 | winning seat, or `NO_PLAYER` |
| `turn_count` | 2 | turns played |
| `current_player` | 3 | turn owner, which during a trade is not the actor |

```python
actors = np.zeros(n, dtype=np.uint8)
status = np.zeros((n, fastcatan.STATUS_INTS), dtype=np.int32)
obs = np.zeros((n, fastcatan.OBS_SIZE), dtype=np.float32)

games.write_actors(actors)
games.write_status(status)
games.write_obs(obs)        # one row per game, from that game's acting seat
```

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

## Observations

`write_obs` encodes a game as `OBS_SIZE` float32 features from one seat's point
of view. The encoding is relative: the chosen seat always occupies slot 0 and
the opponents follow clockwise as +1, +2, +3, so one network serves every seat.

It carries public information plus that seat's own private information and
nothing else. Opponents' resources, development cards by type, and the victory
points hidden in unplayed cards never appear. Two consequences are easy to get
wrong:

- Only the development deck's **size** is exposed. The per-type remainder is not
  public, because it is the complement of what the opponents are holding.
- An unopened trade draft is private to the seat composing it, exactly as in
  `state_view`. The bundles appear once the proposal is open, or when the
  observing seat owns the draft.

Counts are normalized by structural Catan maxima and booleans and one-hot slots
are 0.0 or 1.0. Some values can exceed 1 — a hand spikes past 25 after a
monopoly — because the goal is comparable scale, not a hard bound.

`OBS_FIELDS` maps each field name to `(offset, length)`. The layout, in order:

| Field | Length | Contents |
|---|---|---|
| `players` | 68 | 4 blocks of 17, in relative-seat order (below) |
| `self_private` | 15 | own resources /19, development cards /10, bought this turn /10 |
| `nodes` | 432 | 54 nodes x 4 seats x {settlement, city} |
| `edges` | 288 | 72 edges x 4 seats, road owner |
| `hex_resource` | 114 | 19 hexes x one-hot over 5 resources + desert |
| `hex_number` | 19 | dice number /12 |
| `port_type` | 54 | 9 ports x one-hot over 5 resources + generic |
| `robber` | 19 | robber hex one-hot |
| `phase` | 4 | phase one-hot |
| `flag` | 8 | forced-action flag one-hot |
| `dice_roll` | 13 | one-hot; slot 0 means not yet rolled |
| `turn_count` | 1 | /400 |
| `bank` | 5 | /19 |
| `dev_deck_size` | 1 | cards left in the deck /25 |
| `longest_road_owner` | 5 | `[self, +1, +2, +3, none]` |
| `largest_army_owner` | 5 | `[self, +1, +2, +3, none]` |
| `start_player` | 4 | relative seat one-hot |
| `free_roads` | 1 | /2 |
| `dev_card_played` | 1 | the turn owner already played a card |
| `trade_compose` | 1 | compose budget used /50; 0 unless the seat owns the turn |
| `trade_proposer` | 5 | `[self, +1, +2, +3, none]` |
| `trade_give` | 5 | /19, zero while the draft is private |
| `trade_want` | 5 | /19, zero while the draft is private |
| `trade_response` | 16 | 4 seats x one-hot `[pending, accept, decline, n/a]` |

Each 17-float player block holds victory points /10, hand size /25, development
cards held /10, knights played /10, road length /15, settlements /5, cities /4
and roads /15 remaining, the six port bits, cards owed to a discard /10, whether
the seat owns the turn, and whether it is the seat that must act now. The first
block reports the observing seat's true victory points; the others report only
the publicly visible count.

The encoding is frozen and versioned by `OBS_VERSION`. Any change to a field,
an offset, or a divisor bumps it, and agents trained on an older version must
retrain.

## Event log

`BatchedEnv` can record what happened during a batched step into fixed-width
int32 rows, appended per game from inside the native step loop. Recording is
off by default and costs a single predicted branch per game per step when
disabled.

```python
games = fastcatan.BatchedEnv(n, seed=0)
games.enable_log(fastcatan.log.LOG_TRADES | fastcatan.log.LOG_GAMES,
                 capacity_per_env=64)
games.reset()               # enable before reset to capture the first start

events = np.zeros((4096, fastcatan.EVENT_INTS), dtype=np.int32)
...
count = games.drain_events(events)
assert games.log_overflow == 0
```

| Mask bit | Records |
|---|---|
| `log.LOG_TRADES` | player trades: open, accept, decline, confirm, cancel |
| `log.LOG_GAMES` | game start and game end |
| `log.LOG_ACTIONS` | every `(actor, action)`; with the recorded seed this replays a game exactly |

Event types are `log.GAME_START`, `log.GAME_END`, `log.ACTION`,
`log.TRADE_OPEN`, `log.TRADE_ACCEPT`, `log.TRADE_DECLINE`,
`log.TRADE_CONFIRM`, and `log.TRADE_CANCEL`. `EVENT_FIELDS` maps each field to
`(offset, length)`:

| Field | Offset | Contents |
|---|---|---|
| `env_idx` | 0 | batch slot |
| `game_id` | 1 | episode counter for that slot |
| `step_in_game` | 2 | actions applied before this one |
| `turn` | 3 | turn the event happened on |
| `event_type` | 4 | one of the constants above |
| `actor` | 5 | seat that acted |
| `action` | 6 | triggering action, `-1` for a game start |
| `proposer` | 7 | seat owning the proposal, `-1` outside a trade |
| `partner` | 8 | accepting partner, confirm only |
| `give` | 9..13 | proposer's offered bundle |
| `want` | 14..18 | proposer's requested bundle |
| `public_vp` | 19..22 | `vp_without_dev` per seat, absolute seat order |
| `true_vp` | 23..26 | total victory points per seat |
| `handsize` | 27..30 | resource cards held per seat |
| `winner` | 31 | game end: winning seat, or `NO_PLAYER` |
| `turns` | 32 | game end: turns played |
| `steps` | 33 | game end: actions applied |
| `seed_lo`, `seed_hi` | 34, 35 | game start: the 64-bit episode seed, low word first |

`-1` marks a field this event type does not use. The layout is versioned by
`EVENT_VERSION`.

Four recording rules are worth stating exactly:

- **Context is read before the action runs.** A trade bundle only exists until
  `TRADE_CONFIRM` clears it, so `turn`, `actor`, `proposer`, `give`, `want`,
  and the per-seat arrays all describe the pre-step state. `GAME_END` is the
  exception and reports the finished game, so its tally shows the winner on 10
  victory points rather than 9.
- **A trade action is recorded when its pre-step mask bit is set**, which is
  exactly when the simulator applies it.
- **An accept the simulator could not honor is recorded as the decline that
  happened.** A responder who cannot cover the requested bundle is forced to
  decline; `event_type` reports `TRADE_DECLINE` while `action` keeps the
  `TRADE_ACCEPT` the seat asked for.
- **`LOG_ACTIONS` records every stepped action, mask-illegal ones included.**
  Those are strict no-ops, and replaying them is also a no-op. `NO_ACTION`
  slots record nothing.

Within one step a game emits its action row, then its trade row, then
`GAME_END`, then the `GAME_START` of the game `step_autoreset` put in its
place, whose `game_id` has advanced by one.

`drain_events(out)` copies pending rows in `(env_idx, append order)` and clears
them, returning the row count. Rows that do not fit stay queued, so repeated
drains never lose a record; `pending_events()` sizes the buffer. When a game's
buffer fills, the refused rows are counted in `log_overflow` rather than
dropped silently — a non-zero value means draining less often, or a larger
`capacity_per_env`. The log is not part of a snapshot, and `run_random_games`
raises while recording is enabled because it plays whole games without entering
the step loop.

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
| `OBS_SIZE`, `OBS_FIELDS`, `OBS_PER_PLAYER` | Observation width and field layout |
| `OBS_VERSION` | Observation encoding version |
| `EVENT_INTS`, `EVENT_FIELDS` | Event-row width and field layout |
| `EVENT_VERSION` | Event-row layout version |
| `STATUS_INTS`, `STATUS_FIELDS` | Status-row width and column layout |
| `log` | Event mask bits and event-type constants |

Snapshots are an internal binary format. Only load snapshots created by the
same package version; malformed snapshots are rejected.

## Development

```bash
pip install .
pytest
```

The reference-logger and observation-parity sweeps replay thousands of games
through Python and take about a minute. They run by default; `pytest -m "not
slow"` skips them while iterating.

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

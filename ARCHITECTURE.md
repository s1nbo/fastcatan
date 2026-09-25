# FastCatan architecture

FastCatan contains a rules engine and two simulator interfaces: one game and a
contiguous batch of independent games. It has no player implementations,
decision-making logic, outcome scoring, or numeric feature encoder.

## Source ownership

| Path | Responsibility |
|---|---|
| `include/state.hpp` | Game state, board layout, phases, flags, and constants |
| `include/topology.hpp` | Static board adjacency tables |
| `include/rng.hpp` | Per-game deterministic random-number generator |
| `include/rules.hpp` | Action IDs, sampled transitions, and exact physical outcomes |
| `include/mask.hpp` | Legal-action bitset contract |
| `include/batched_env.hpp` | Contiguous batched simulator storage |
| `src/catan/rules.cpp` | Setup, legality, transitions, trades, awards, and chance events |
| `src/catan/batched_env.cpp` | Batched reset, stepping, snapshots, RNG control, and masks |
| `bindings/pycatan/bindings.cpp` | Validated Python and NumPy boundary |

`GameState` owns all mutable state, including its random-number generator and
cached legal-action mask. `BoardLayout` is randomized during reset and remains
fixed for the life of the game. Both are trivially copyable so snapshots can
round-trip them exactly within one package version.

## Transition contract

`reset_one` initializes a board, pieces, bank, development deck, turn owner,
random-number state, and legal-action mask. `step_one` applies one action and
returns whether the game is terminal. An out-of-range or currently illegal
action is a strict no-op. The legal-action mask is rebuilt after every accepted
transition.

`enumerate_action_outcomes` is the non-sampling form of the same transition
model. It enumerates dice, development-card, and robber-steal results from the
physical state. It does not choose actions or assign values to outcomes.

`current_player` owns the turn. `actor_to_act(GameState)` owns the next decision;
these differ while another seat discards after a seven.

Games normally end when a player reaches ten victory points. `MAX_TURNS` is a
backstop for non-progressing games, and `MAX_TRADE_COMPOSE_PER_TURN` prevents a
single turn from cycling forever while composing and cancelling trades.

## Python boundary

`Env` exposes semantic board and player fields, legal actions, stepping, exact
outcomes, and validated snapshots. `state_view` filters private state by player
without converting it into a numeric feature representation. The API rejects
use before reset and checks every seat, resource, card, node, edge, hex, and port
index before entering the C++ arrays.

`BatchedEnv` owns aligned arrays of states and layouts. Its hot loops release the
Python lock and use OpenMP when available. Normal steps preserve terminal states.
Automatic terminal reset is an explicit continuous-simulation operation. Bulk
snapshot, restore, and RNG reseeding support replay and independent batch slots.

## Verification

Python tests cover deterministic replay, illegal-action no-ops, action-mask
integrity, official-rule scenarios, trade flow, input validation, snapshots,
and batched operation. The standalone `fuzz_invariants` executable runs the
same conservation and range invariants over a much larger number of games.

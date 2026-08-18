# FastCatan architecture

FastCatan 2.x is a simulator library. It owns Catan state, legal transitions,
observations, deterministic randomness, batching, Python bindings, and simulator
conformance tests. It does not own agents, training loops, planning algorithms,
or tournament evaluation; those live in the separate `catan-rl` repository.

## Core ownership

| path | responsibility |
|---|---|
| `include/state.hpp` | `GameState`, `BoardLayout`, phases, flags, acting-seat authority |
| `include/rules.hpp` | action IDs, reset/step contract, weighted chance expansion |
| `include/obs.hpp` | versioned POV and privileged diagnostic observation schemas |
| `include/batched_env.hpp` | vectorized simulator storage and branching primitives |
| `src/catan/rules.cpp` | legal masks, rule transitions, awards, trades, chance outcomes |
| `src/catan/obs.cpp` | information-set-safe POV encoding and full diagnostic encoding |
| `src/catan/batched_env.cpp` | batched reset, step, snapshot, observation, and signatures |
| `bindings/pycatan/bindings.cpp` | allocation-free NumPy-facing Python API |

The single acting-seat authority is `actor_to_act(GameState)`. `current_player`
is the turn owner and is not necessarily the decision owner during cross-seat
forced phases such as discarding after a seven.

## Public boundary

`Env` provides single-game reset, legal mask, step, observation, snapshot, RNG,
and read-only state inspection. `BatchedEnv` provides the same simulator
operations over contiguous state arrays, including raw non-resetting steps for
branching consumers.

`write_obs` is the deployable partial-information view. `write_obs_full` is an
explicitly privileged diagnostic view and must not be fed to decentralized
policies. Observation meaning is guarded by `OBS_SEMANTICS_VERSION` even when
tensor width is unchanged.

Weighted `expand_action` is a transition-model primitive: it enumerates dice,
development-card, and robber outcomes without choosing actions. Search policy
and value logic does not belong in this repository.

## Build and verification

The C++ core is compiled into a nanobind stable-ABI module through
scikit-build-core. OpenMP is optional and affects throughput only. The Python
package has only NumPy as a runtime dependency.

Simulator tests cover deterministic replay, mask integrity, illegal-action
no-ops, official-rule scenarios, information noninterference, snapshots, and
batched/single-environment parity. Cross-engine and agent evaluation suites are
owned by `catan-rl`.

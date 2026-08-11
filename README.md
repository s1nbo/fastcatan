# fastcatan

[![PyPI](https://img.shields.io/pypi/v/fastcatan.svg)](https://pypi.org/project/fastcatan/)
[![Python](https://img.shields.io/pypi/pyversions/fastcatan.svg)](https://pypi.org/project/fastcatan/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **high-throughput Settlers of Catan simulator** — C++ core with Python bindings
([nanobind](https://github.com/wjakob/nanobind)), built for reinforcement-learning
research. Millions of environment steps per second, a flat observation vector, a
legal-action bitmask, and a batched vectorized environment for GPU training.

- ⚡ **Fast** — pure-C++ rules engine, ~4.6M batched env-steps/s on a single desktop.
- 🧠 **RL-ready** — fixed-size `float32` observation, `uint64` action mask, scalar reward.
- 📦 **Batched** — `BatchedEnv` steps thousands of games per call (OpenMP, GIL released).
- 🎯 **Deterministic** — seeded games are fully reproducible.

> **Stability:** the rules engine and the observation/action interface are frozen
> and stable — pin an exact version (e.g. `fastcatan==1.0.2`).

## Install

```bash
pip install fastcatan
```

Prebuilt wheels: **Linux x86_64** and **macOS (Apple Silicon)**, CPython **3.12+**
(one `abi3` wheel serves 3.12 / 3.13 / 3.14). Other platforms build from the sdist
automatically (needs a C++ toolchain + CMake ≥ 3.27).

## Quickstart

Play one game with a random legal policy:

```python
import numpy as np
import fastcatan

env = fastcatan.Env()
env.reset(seed=0)

mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)   # legal-action bitmask
obs  = np.zeros(fastcatan.OBS_SIZE,  dtype=np.float32)   # observation buffer
rng  = np.random.default_rng(0)

while True:
    env.action_mask(mask)                 # fill legal moves for current player
    env.write_obs(env.current_player, obs)   # fill that player's POV observation
    legal = np.flatnonzero(np.unpackbits(mask.view(np.uint8), bitorder="little")
                           [:fastcatan.NUM_ACTIONS])
    action = int(rng.choice(legal))
    reward, done = env.step(action)
    if done:
        vps = [env.player_vp(p) for p in range(fastcatan.NUM_PLAYERS)]
        winner = next((p for p, v in enumerate(vps) if v >= 10), -1)
        print("winner:", winner, "vps:", vps)
        break
```

### Batched (vectorized) environment

For RL throughput, step many games at once and drive them with a batched policy:

```python
import numpy as np, fastcatan

N = 4096
env = fastcatan.BatchedEnv(N, seed=0)
env.reset()

masks = np.zeros((N, fastcatan.MASK_WORDS), dtype=np.uint64)
obs   = np.zeros((N, fastcatan.OBS_SIZE),  dtype=np.float32)
acts  = np.zeros(N, dtype=np.uint32)
rew   = np.zeros(N, dtype=np.float32)
done  = np.zeros(N, dtype=np.uint8)

env.write_masks(masks)      # all N legal masks in one C++ call
env.write_obs(obs)          # all N current-player observations
# ... your batched policy fills `acts` ...
env.step_raw(acts, rew, done)   # steps all N games (OpenMP, GIL released)
```

## API at a glance

| symbol | meaning |
|--------|---------|
| `Env` | single-game environment (`reset`, `step`, `action_mask`, `write_obs`, …) |
| `BatchedEnv` | vectorized environment over N games (`step_raw`, `write_masks`, `write_obs`, …) |
| `OBS_SIZE` / `OBS_FULL_SIZE` | observation length (POV / full-information) |
| `NUM_ACTIONS` | size of the action space |
| `MASK_WORDS` | `uint64` words in a legal-action bitmask (`ceil(NUM_ACTIONS/64)`) |
| `NUM_PLAYERS`, `NUM_NODES`, `NUM_EDGES`, `NUM_HEXES`, `NUM_PORTS` | board constants |
| `SKIP_ACTION` | no-op action id (parked finished games in a batch) |

`env.step(action)` returns `(reward, done)`. Observations are written into
caller-provided buffers (no per-step allocation). See [`examples/`](examples/) for
random and alpha-beta players.

## Development

Build from source (tunes `-march=native` for your machine):

```bash
pip install .
pytest tests/
```

## License

[MIT](LICENSE) © 2026 s1nbo

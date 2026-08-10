# fastcatan — frozen Catan simulator

High-throughput Catan simulator (C++ core, nanobind Python bindings). Builds the
`fastcatan` Python package. **This repo is frozen at v1.0.0** — the simulator is
stable and no longer under active development. AI / RL work lives in a separate
repository and consumes this as a pinned dependency.

## Layout

| path | what |
|------|------|
| `src/`, `include/` | C++ simulator core |
| `bindings/pycatan/bindings.cpp` | nanobind bindings |
| `python/fastcatan/` | Python package (wraps the compiled extension) |
| `sim/`, `examples/` | helpers, usage examples |
| `tests/` | pytest + C++ fuzz/invariant tests |
| `CMakeLists.txt`, `pyproject.toml` | build (scikit-build-core) |

## Install

From PyPI (prebuilt abi3 wheel, no toolchain needed, CPython 3.12+):

```bash
pip install fastcatan
```

Downstream projects pin an exact version: `fastcatan==1.0.0`.

From source (needs a C++ toolchain + CMake ≥ 3.27) — dev/editable or non-Linux:

```bash
pip install .            # tunes -march=native for THIS machine (fast local build)
```

## Publishing to PyPI

Publishing is automated: pushing a version tag (`v1.0.0`) runs
`.github/workflows/publish.yml`, which builds one **abi3** wheel (nanobind
`STABLE_ABI` → one wheel serves 3.12/3.13/3.14) plus an sdist via
[cibuildwheel](https://cibuildwheel.pypa.io) and uploads through PyPI
**Trusted Publishing** (OIDC, no secrets). One-time setup on PyPI:
configure the trusted publisher for project `fastcatan` (repo + workflow
`publish.yml` + environment `pypi`).

The published wheel is built portable (`-march=x86-64-v2`, runs on any 2009+ CPU)
inside a `manylinux_2_28` container (GCC 12+, required for C++23). Local builds
keep `-march=native`. Build a wheel by hand:

```bash
python -m build --wheel                               # native (this CPU)
python -m build --wheel -Ccmake.define.FASTCATAN_ARCH=x86-64-v2   # portable
```

Currently **Linux x86_64** only; add macOS/Windows to the CI `archs` matrix later.

## Test

```bash
pip install .
pytest tests/
```

## Versioning

Frozen releases are git tags (`v1.0.0`, …). Bump `pyproject.toml` `version` and
tag if the simulator is ever revised; downstream projects pin an exact tag.

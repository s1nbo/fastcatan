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

Editable / from source (needs a C++ toolchain + CMake ≥ 3.27):

```bash
pip install .
```

Pin a frozen version from another project:

```bash
pip install "fastcatan @ git+https://<host>/<you>/catan-sim.git@v1.0.0"
```

## Build a distributable wheel

```bash
python -m build --wheel      # -> dist/fastcatan-1.0.0-cp3XX-...whl
```

The wheel is platform + Python-version specific (compiled extension).

## Test

```bash
pip install .
pytest tests/
```

## Versioning

Frozen releases are git tags (`v1.0.0`, …). Bump `pyproject.toml` `version` and
tag if the simulator is ever revised; downstream projects pin an exact tag.

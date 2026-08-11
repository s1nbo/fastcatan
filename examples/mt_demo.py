"""How the multithreading actually works — a visual walkthrough.

Run it and watch the CPU numbers:

    python mt_demo.py

Three demos, in order:

  1. Proof that the GIL is released. The same 4 threads pin 1 core when they
     run python bytecode and 4 cores when they run BatchedEnv calls.
  2. Shard ladder. Random-agent rollouts at 1, 2, 4, ... threads, showing
     cores busy and env-steps/s, so you can see where scaling dies.
  3. The winner, re-run with a live CPU meter.

"Cores busy" = process CPU time / wall time, from os.times(). 8.0 means eight
cores were saturated for the whole interval. Compare it to `cpus` in the
header.
"""
from __future__ import annotations

import os
import sys
import threading
import time

# libgomp reads this at init, so it must be set before fastcatan is imported.
# Pinned to 1 on purpose: this demo parallelises with python threads, and the
# two levels only fight each other (see mt_random_test.py for the numbers).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import fastcatan as fc  # noqa: E402

NCPU = os.cpu_count() or 1
ENVS_PER_SHARD = 4096   # below ~4k the numpy work per call is too small to overlap
STEPS = 150
# Fresh envs sit in the cheap setup phase for their first ~60 steps, which
# flatters throughput. Burn those before starting the clock.
WARMUP = 80

# Bit-select tables: POPC[b] = set bits in byte b, SEL[b, k] = index of b's
# k-th set bit. Turns "pick a uniform legal action" into integer ops on the
# 40 mask bytes instead of an unpacked (n, 286) matrix.
POPC = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int32)
SEL = np.zeros((256, 8), dtype=np.uint8)
for _b in range(256):
    _s = [i for i in range(8) if _b >> i & 1]
    SEL[_b, :len(_s)] = _s


def cpu_seconds() -> float:
    """CPU time burned by this process across ALL its threads."""
    t = os.times()
    return t.user + t.system


class Timer:
    """Wall time + cores-busy over a block."""

    def __enter__(self):
        self.w0, self.c0 = time.perf_counter(), cpu_seconds()
        return self

    def __exit__(self, *_):
        self.wall = time.perf_counter() - self.w0
        self.cores = (cpu_seconds() - self.c0) / self.wall


class Shard:
    """One BatchedEnv + its buffers. One of these per python thread."""

    def __init__(self, n: int, seed: int):
        self.n = n
        self.env = fc.BatchedEnv(n, seed)
        self.env.reset()
        self.rng = np.random.default_rng(seed)
        self.masks = np.zeros((n, fc.MASK_WORDS), dtype=np.uint64)
        self.mb = self.masks.view(np.uint8)     # (n, 40) alias, no copy
        self.rows = np.arange(n)
        self.obs = np.zeros((n, fc.OBS_SIZE), dtype=np.float32)
        self.acts = np.zeros(n, dtype=np.uint32)
        self.rew = np.zeros(n, dtype=np.float32)
        self.done = np.zeros(n, dtype=np.uint8)
        self.games = 0

    def random_actions(self) -> None:
        cnt = POPC[self.mb]
        cum = np.cumsum(cnt, axis=1)
        total = cum[:, -1]
        k = (self.rng.random(self.n) * total).astype(np.int32)
        b = (cum > k[:, None]).argmax(1)
        r = self.rows
        self.acts[:] = b * 8 + SEL[self.mb[r, b], k - (cum[r, b] - cnt[r, b])]
        if not total.all():
            self.acts[total == 0] = fc.SKIP_ACTION

    def rollout(self, steps: int) -> None:
        """The real workload: mask -> policy -> step, repeat."""
        for _ in range(steps):
            self.env.write_masks(self.masks)   # GIL released
            self.random_actions()              # pure python/numpy
            self.env.step(self.acts, self.rew, self.done)  # GIL released
            self.games += int(np.count_nonzero(self.done))

    def spin_obs(self, iters: int) -> None:
        """C++-only work: write_obs touches no python object while it runs."""
        for _ in range(iters):
            self.env.write_obs(self.obs)


def run_threads(fn, args_per_thread) -> Timer:
    """Start one thread per item, join them all, report wall + cores."""
    threads = [threading.Thread(target=fn, args=a) for a in args_per_thread]
    with Timer() as t:
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    return t


# --------------------------------------------------------------------------
# Demo 1 — where the GIL does and does not stand in the way
# --------------------------------------------------------------------------
def demo_gil(nthreads: int = 4) -> None:
    print(f"\n=== 1. GIL proof — {nthreads} threads, same count each ===")
    print("Identical thread setup. Only the work inside differs.\n")

    def python_spin(n):
        x = 0
        for i in range(n):     # bytecode: holds the GIL
            x += i * i
        return x

    t = run_threads(python_spin, [(6_000_000,)] * nthreads)
    print(f"  python bytecode  wall {t.wall:5.2f}s   cores busy {t.cores:4.1f}"
          "   <- serialised by the GIL")

    shards = [Shard(ENVS_PER_SHARD, 100 + i) for i in range(nthreads)]
    iters = 400
    t = run_threads(Shard.spin_obs, [(s, iters) for s in shards])
    print(f"  BatchedEnv C++   wall {t.wall:5.2f}s   cores busy {t.cores:4.1f}"
          "   <- GIL released, real parallelism")
    print("\n  Every hot BatchedEnv call (write_masks / write_obs / step) does")
    print("  `nb::gil_scoped_release` before touching C++, so other python")
    print("  threads keep running. That is the whole trick.")


# --------------------------------------------------------------------------
# Demo 2 — how far it scales with the real random-agent loop
# --------------------------------------------------------------------------
def demo_ladder() -> list[tuple[int, float, float]]:
    print(f"\n=== 2. Shard ladder — {ENVS_PER_SHARD:,} envs per thread,"
          f" {STEPS} steps ===")
    print("Each thread owns its own BatchedEnv, so there is nothing to lock.\n")
    print(f"  {'threads':>7} {'envs':>9} {'wall s':>7} {'cores':>6} "
          f"{'steps/s':>12} {'speedup':>8}")

    counts, results, base = [], [], None
    c = 1
    while c <= NCPU:
        counts.append(c)
        c *= 2
    for n in counts:
        shards = [Shard(ENVS_PER_SHARD, 1000 * i) for i in range(n)]
        run_threads(Shard.rollout, [(s, WARMUP) for s in shards])
        t = run_threads(Shard.rollout, [(s, STEPS) for s in shards])
        sps = ENVS_PER_SHARD * n * STEPS / t.wall
        base = base or sps
        results.append((n, sps, t.cores))
        print(f"  {n:>7} {ENVS_PER_SHARD * n:>9,} {t.wall:>7.2f} {t.cores:>6.1f} "
              f"{sps:>12,.0f} {sps / base:>7.2f}x")
        del shards
    return results


# --------------------------------------------------------------------------
# Demo 3 — the best config, with a live meter
# --------------------------------------------------------------------------
def demo_live(nthreads: int, rate: float, seconds: float = 6.0) -> None:
    print(f"\n=== 3. Live run — {nthreads} threads,"
          f" {ENVS_PER_SHARD * nthreads:,} envs, ~{seconds:.0f}s ===\n")
    shards = [Shard(ENVS_PER_SHARD, 7000 + 13 * i) for i in range(nthreads)]
    stop = threading.Event()

    def meter():
        """Sampled from the main thread's own loop — itself proof the
        interpreter stays responsive while the workers are in C++."""
        w0, c0 = time.perf_counter(), cpu_seconds()
        while not stop.wait(0.25):
            w1, c1 = time.perf_counter(), cpu_seconds()
            cores = (c1 - c0) / (w1 - w0)
            w0, c0 = w1, c1
            filled = int(round(cores / NCPU * 40))
            bar = "#" * filled + "." * (40 - filled)
            sys.stdout.write(f"\r  [{bar}] {cores:5.1f} / {NCPU} cores")
            sys.stdout.flush()

    run_threads(Shard.rollout, [(s, WARMUP) for s in shards])
    # `rate` is batched steps/s per thread, measured by the ladder above.
    steps = max(50, int(seconds * rate))
    mt = threading.Thread(target=meter, daemon=True)
    mt.start()
    t = run_threads(Shard.rollout, [(s, steps) for s in shards])
    stop.set()
    mt.join()

    total = ENVS_PER_SHARD * nthreads * steps
    print(f"\r{' ' * 70}\r  wall {t.wall:.2f}s   cores busy {t.cores:.1f} / {NCPU}"
          f"   {total / t.wall:,.0f} env-steps/s")
    print(f"  {total:,} env-steps, {sum(s.games for s in shards):,} games finished")


def main() -> int:
    print(f"fastcatan  {fc.__file__}")
    print(f"cpus={NCPU}  OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}  "
          f"envs/shard={ENVS_PER_SHARD:,}")

    demo_gil()
    results = demo_ladder()
    best = max(results, key=lambda r: r[1])
    print(f"\n  best: {best[0]} threads -> {best[1]:,.0f} env-steps/s "
          f"at {best[2]:.1f} cores")
    per_thread_rate = best[1] / (best[0] * ENVS_PER_SHARD)  # batched steps/s
    demo_live(best[0], per_thread_rate)

    print("\n--- how to max out your CPU ---")
    print(f"  * {best[0]} python threads, one BatchedEnv each, "
          f"{ENVS_PER_SHARD:,}+ envs per thread")
    print("  * OMP_NUM_THREADS=1 — OpenMP inside step() only speeds up the ~3%")
    print("    of the loop that is C++; it steals cores from the shards")
    print("  * skip write_obs unless the policy reads it (1.6ms/step at 4096)")
    print("  * shards below ~4k envs leave cores idle: the numpy work per call")
    print("    gets too short to overlap with the other threads")
    print("  * many small shards beat one huge BatchedEnv even at equal total")
    print("    envs — 8x4096 stays in cache, 1x32768 goes memory-bound")
    print(f"\n  the ceiling is the GIL, not the env: cores stop at ~{best[2]:.0f}"
          f" of {NCPU} because")
    print("  the numpy policy between the two C++ calls still needs the lock.")
    print("  past that, move the policy to torch/GPU or shard across processes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

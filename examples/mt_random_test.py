"""Multi-threaded random-agent throughput test for BatchedEnv.

Two independent levels of parallelism, combinable:

  1. OpenMP inside the C++ step (``--omp N``). One BatchedEnv, N OpenMP
     workers splitting the env loop. OMP_NUM_THREADS must be set before
     libgomp initialises, so it is set here before ``import fastcatan``.
  2. Python threads over shards (``--shards N``). N BatchedEnv instances,
     one per threading.Thread. Legal because every hot BatchedEnv call
     releases the GIL, so the policy numpy work overlaps too.

Measured on this box at N=4096 (per batched step, single thread):

    step         0.10 ms   <- the C++ env is not the bottleneck
    write_masks  0.01 ms
    write_obs    1.61 ms   <- skipped unless --obs; a random agent never reads it
    sampling     0.62 ms   (bytewise bit-select; the naive (N, NUM_ACTIONS)
                            float-noise argmax costs 1.82 ms for the same result)

Only ~7.8 of 286 actions are legal in a typical state, so the sampler works
on the 40 mask bytes rather than on an unpacked (N, 286) matrix.

Because the policy dominates, python threads beat OpenMP here and the two
do not compose: on this 24-core box, ``--envs 32768 --shards 8`` reaches
10.3M env-steps/s, while adding ``--omp`` on top of that only costs
contention. Give each shard >= ~4k envs or the per-call numpy work gets too
small to overlap.

Usage:
    python mt_random_test.py                  # sweep 1/2/4/.. threads, both modes
    python mt_random_test.py --omp 8          # 8 OpenMP workers
    python mt_random_test.py --shards 8       # 8 python-thread shards
    python mt_random_test.py --shards 4 --omp 2 --obs
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", type=int, default=4096, help="total envs across all shards")
    p.add_argument("--steps", type=int, default=200, help="batched steps per env")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--omp", type=int, help="OpenMP workers for one measured run")
    p.add_argument("--shards", type=int, help="python-thread shards for one measured run")
    p.add_argument("--obs", action="store_true",
                   help="also call write_obs each step (a real policy would)")
    p.add_argument("--max-threads", type=int, default=os.cpu_count() or 1,
                   help="upper bound for the sweep")
    return p.parse_args(argv)


ARGS = parse_args()

# Must precede the fastcatan import: libgomp reads OMP_NUM_THREADS at init.
# Default 1 so a --shards run never oversubscribes; --omp with --shards is
# the caller explicitly asking for both levels at once.
_OMP = ARGS.omp if ARGS.omp is not None else 1
os.environ["OMP_NUM_THREADS"] = str(_OMP)

import numpy as np  # noqa: E402
import fastcatan as fc  # noqa: E402

A = fc.NUM_ACTIONS
MASK_BYTES = fc.MASK_WORDS * 8

# POPC[b] = set bits in byte b. SEL[b, k] = index of b's k-th set bit (LSB
# first). Both tiny and built once; they turn "pick a uniform set bit out of
# MASK_BYTES bytes" into a handful of (n, MASK_BYTES) integer ops.
POPC = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int32)
SEL = np.zeros((256, 8), dtype=np.uint8)
for _b in range(256):
    _set = [i for i in range(8) if _b >> i & 1]
    SEL[_b, :len(_set)] = _set


class Shard:
    """One BatchedEnv plus its pre-allocated buffers and RNG."""

    def __init__(self, n: int, seed: int):
        self.n = n
        self.env = fc.BatchedEnv(n, seed)
        self.env.reset()
        self.rng = np.random.default_rng(seed)
        self.masks = np.zeros((n, fc.MASK_WORDS), dtype=np.uint64)
        self.mask_bytes = self.masks.view(np.uint8)  # (n, MASK_BYTES) alias
        self.rows = np.arange(n)
        self.obs = np.zeros((n, fc.OBS_SIZE), dtype=np.float32)
        self.acts = np.zeros(n, dtype=np.uint32)
        self.rew = np.zeros(n, dtype=np.float32)
        self.done = np.zeros(n, dtype=np.uint8)
        self.games = 0
        self.wins = np.zeros(4, dtype=np.int64)
        self.env_seconds = 0.0
        self.illegal = 0

    def random_actions(self) -> None:
        """Uniform random legal action per env, vectorised over the bitset.

        The mask is (n, MASK_WORDS) uint64; bit i of word w is action
        w*64 + i, so viewing it as uint8 makes byte j hold actions
        8j..8j+7 with the low bit first. Pick k uniformly in
        [0, popcount), find the byte holding the k-th set bit by cumsum,
        then index the k'-th set bit inside it via SEL. Padding bits above
        NUM_ACTIONS are never set by the C++ side, so no trim is needed.
        """
        cnt = POPC[self.mask_bytes]                        # (n, MASK_BYTES)
        cum = np.cumsum(cnt, axis=1)
        total = cum[:, -1]
        k = (self.rng.random(self.n) * total).astype(np.int32)
        byte_idx = (cum > k[:, None]).argmax(1)            # first byte past k
        r = self.rows
        k_in = k - (cum[r, byte_idx] - cnt[r, byte_idx])   # rank within that byte
        self.acts[:] = byte_idx * 8 + SEL[self.mask_bytes[r, byte_idx], k_in]
        dead = total == 0
        if dead.any():
            self.acts[dead] = fc.SKIP_ACTION
            self.illegal += self.check_legal(~dead)
        else:
            self.illegal += self.check_legal(None)

    def check_legal(self, alive) -> int:
        """Count emitted actions whose bit is NOT set in their own mask."""
        a = self.acts if alive is None else self.acts[alive]
        mb = self.mask_bytes if alive is None else self.mask_bytes[alive]
        bit = (mb[np.arange(len(a)), a >> 3] >> (a & 7).astype(np.uint8)) & 1
        return int(np.count_nonzero(bit == 0))

    def run(self, steps: int) -> None:
        want_obs = ARGS.obs
        for _ in range(steps):
            t0 = time.perf_counter()
            self.env.write_masks(self.masks)
            if want_obs:
                self.env.write_obs(self.obs)
            self.env_seconds += time.perf_counter() - t0

            self.random_actions()  # stand-in for the batched policy

            t0 = time.perf_counter()
            self.env.step(self.acts, self.rew, self.done)  # auto-resets on done
            self.env_seconds += time.perf_counter() - t0

            fin = np.flatnonzero(self.done)
            if fin.size:
                self.games += int(fin.size)
                for i in fin:
                    w = self.env.last_winner(int(i))
                    if w < 4:
                        self.wins[w] += 1


def measure(total_envs: int, steps: int, shards: int, seed: int) -> dict:
    per = total_envs // shards
    if per == 0:
        raise SystemExit(f"--envs {total_envs} too small for {shards} shards")
    objs = [Shard(per, seed + 1000 * k) for k in range(shards)]

    t0 = time.perf_counter()
    if shards == 1:
        objs[0].run(steps)
    else:
        threads = [threading.Thread(target=s.run, args=(steps,)) for s in objs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    wall = time.perf_counter() - t0

    env_steps = per * shards * steps
    return {
        "wall": wall,
        "env_seconds": sum(s.env_seconds for s in objs) / shards,  # mean per shard
        "steps": env_steps,
        "sps": env_steps / wall,
        "games": sum(s.games for s in objs),
        "wins": sum(s.wins for s in objs),
        "illegal": sum(s.illegal for s in objs),
    }


def sweep() -> int:
    """Re-exec self once per configuration; OMP_NUM_THREADS needs a fresh process."""
    counts = [1]
    while counts[-1] * 2 <= ARGS.max_threads:
        counts.append(counts[-1] * 2)
    if counts[-1] != ARGS.max_threads:
        counts.append(ARGS.max_threads)

    base = [sys.executable, __file__, "--envs", str(ARGS.envs),
            "--steps", str(ARGS.steps), "--seed", str(ARGS.seed)]
    if ARGS.obs:
        base.append("--obs")
    print(f"fastcatan {fc.__file__}")
    print(f"envs={ARGS.envs} steps={ARGS.steps} obs={ARGS.obs} cpus={os.cpu_count()}\n")
    for mode, flag in (("openmp", "--omp"), ("py-threads", "--shards")):
        print(f"--- {mode} ---")
        print(f"{'threads':>7} {'wall s':>8} {'steps/s':>12} {'speedup':>8}")
        first = None
        for t in counts:
            if mode == "py-threads" and ARGS.envs // t == 0:
                continue
            out = subprocess.run(base + [flag, str(t)], capture_output=True, text=True)
            if out.returncode != 0:
                print(out.stdout + out.stderr)
                return 1
            wall, sps = (float(x) for x in out.stdout.strip().split()[:2])
            first = first or sps
            print(f"{t:>7} {wall:>8.2f} {sps:>12,.0f} {sps / first:>7.2f}x")
        print()
    return 0


def main() -> int:
    if ARGS.omp is None and ARGS.shards is None:
        return sweep()

    shards = ARGS.shards or 1
    r = measure(ARGS.envs, ARGS.steps, shards, ARGS.seed)
    if r["illegal"]:
        print(f"ILLEGAL ACTIONS: {r['illegal']}", file=sys.stderr)
        return 1
    # Machine-readable first line for the sweep; human summary on stderr.
    print(f"{r['wall']:.6f} {r['sps']:.1f}")
    print(f"# omp={_OMP} shards={shards} envs={ARGS.envs} steps={ARGS.steps} "
          f"obs={ARGS.obs} env_steps={r['steps']:,} games={r['games']} "
          f"wins={r['wins'].tolist()} env_time={r['env_seconds']:.2f}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

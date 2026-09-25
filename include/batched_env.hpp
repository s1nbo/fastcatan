#pragma once
#include <cstdint>
#include "state.hpp"

namespace catan {

    // BatchedEnv — N independent (GameState, BoardLayout) pairs in contiguous
    // buffers. Loops use OpenMP when available.
    //
    struct BatchedEnv {
        uint32_t n;                  // number of envs
        GameState*  states;          // aligned array, length n
        BoardLayout* layouts;        // length n
        uint64_t    seed_counter;    // monotonic per-env seed derivation
        uint8_t*    last_winner;     // last game's winner per env, NO_PLAYER if none
    };

    // Allocate and zero buffers. Does NOT call reset_one; the caller invokes
    // batched_env_reset. Returns false if any allocation fails.
    bool batched_env_init(BatchedEnv& env, uint32_t n_envs,
                          uint64_t master_seed) noexcept;

    // Free buffers. Safe to call on a zero-initialized BatchedEnv.
    void batched_env_destroy(BatchedEnv& env) noexcept;

    // Reset all envs to fresh starting positions, each with a unique seed.
    void batched_env_reset(BatchedEnv& env) noexcept;

    // Step every env by one action without resetting terminal states.
    // `actions` and `dones_out` both have length n. NO_ACTION leaves a slot
    // unchanged while still reporting whether its current state is terminal.
    void batched_env_step(BatchedEnv& env,
                          const uint32_t* actions,
                          uint8_t* dones_out) noexcept;

    // Continuous-rollout convenience: step every slot, record the winner, and
    // immediately reset each terminal game with a fresh deterministic seed.
    void batched_env_step_autoreset(BatchedEnv& env,
                                    const uint32_t* actions,
                                    uint8_t* dones_out) noexcept;

    // Write legal-action mask for every env. `out` length n*MASK_WORDS.
    void batched_env_write_masks(const BatchedEnv& env, uint64_t* out) noexcept;

    // Explicit no-action sentinel for independently progressing batch slots.
    inline constexpr uint32_t NO_ACTION = 0xFFFFFFFFu;

    // Serialize/restore every slot using rows of SNAPSHOT_BYTES. Restoring
    // recomputes legal masks; callers must validate untrusted bytes first.
    void batched_env_save(const BatchedEnv& env, uint8_t* out) noexcept;
    void batched_env_load(BatchedEnv& env, const uint8_t* data) noexcept;

    // Replace each slot's RNG state without changing its game state.
    void batched_env_reseed(BatchedEnv& env, const uint64_t* seeds) noexcept;
}  // namespace catan

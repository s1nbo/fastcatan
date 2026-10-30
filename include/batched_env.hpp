#pragma once
#include <cstdint>
#include "event_log.hpp"
#include "state.hpp"

namespace catan {

    // BatchedEnv — N independent (GameState, BoardLayout) pairs in contiguous
    // buffers. Loops use OpenMP when available.
    struct BatchedEnv {
        uint32_t n;                  // number of envs
        GameState*  states;          // aligned array, length n
        BoardLayout* layouts;        // length n
        uint64_t    seed_counter;    // monotonic per-env seed derivation
        uint8_t*    last_winner;     // last game's winner per env, NO_PLAYER if none

        // --- Per-env episode bookkeeping (not part of a snapshot) ---
        uint32_t*   game_id;         // episode counter; 0 after reset, ++ per autoreset
        uint32_t*   step_in_game;    // actions applied in the current episode
        uint64_t*   game_seed;       // seed the current episode was reset with

        EventLog    log;             // opt-in event recording; disabled by default
    };

    // Allocate and zero buffers. Does NOT call reset_one; the caller invokes
    // batched_env_reset. Returns false if any allocation fails.
    bool batched_env_init(BatchedEnv& env, uint32_t n_envs,
                          uint64_t master_seed) noexcept;

    // Free buffers. Safe to call on a zero-initialized BatchedEnv.
    void batched_env_destroy(BatchedEnv& env) noexcept;

    // Reset all envs to fresh starting positions, each with a unique seed.
    // Starts episode 0 in every slot and emits a game-start event when
    // LOG_GAMES is recorded, so enable logging before resetting to capture the
    // first episode.
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

    // Play exactly `num_games` fresh games with uniformly random legal actions.
    // Work is distributed across the batch slots and remains entirely native.
    // Returns the number of completed games and writes the total action count.
    // Bypasses the batched step loop, so it records no events; callers must
    // disable logging first.
    uint64_t batched_env_run_random_games(BatchedEnv& env,
                                          uint64_t num_games,
                                          uint64_t& total_steps) noexcept;

    // Write legal-action mask for every env. `out` length n*MASK_WORDS.
    void batched_env_write_masks(const BatchedEnv& env, uint64_t* out) noexcept;

    // Write actor_to_act for every env. `out` length n.
    void batched_env_write_actors(const BatchedEnv& env, uint8_t* out) noexcept;

    // Per-env status row: done, winner (NO_PLAYER if none), turn_count,
    // current_player. `out` length n*STATUS_INTS.
    inline constexpr uint32_t STATUS_DONE           = 0;
    inline constexpr uint32_t STATUS_WINNER         = 1;
    inline constexpr uint32_t STATUS_TURN_COUNT     = 2;
    inline constexpr uint32_t STATUS_CURRENT_PLAYER = 3;
    inline constexpr uint32_t STATUS_INTS           = 4;
    void batched_env_write_status(const BatchedEnv& env, int32_t* out) noexcept;

    // Observations, one row of OBS_SIZE floats per env. `write_obs` uses each
    // env's own actor_to_act as the point of view; `write_obs_pov` takes one
    // seat per env. Seats must be < NUM_PLAYERS.
    void batched_env_write_obs(const BatchedEnv& env, float* out) noexcept;
    void batched_env_write_obs_pov(const BatchedEnv& env, const uint8_t* povs,
                                   float* out) noexcept;

    // Explicit no-action sentinel for independently progressing batch slots.
    inline constexpr uint32_t NO_ACTION = 0xFFFFFFFFu;

    // Serialize/restore every slot using rows of SNAPSHOT_BYTES. Restoring
    // recomputes legal masks; callers must validate untrusted bytes first.
    // Episode bookkeeping and the event log are not part of a snapshot.
    void batched_env_save(const BatchedEnv& env, uint8_t* out) noexcept;
    void batched_env_load(BatchedEnv& env, const uint8_t* data) noexcept;

    // Replace each slot's RNG state without changing its game state.
    void batched_env_reseed(BatchedEnv& env, const uint64_t* seeds) noexcept;
}  // namespace catan

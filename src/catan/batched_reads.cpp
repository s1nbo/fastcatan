// Batched read-only exports: one row per env written straight into a
// caller-provided buffer, so an RL loop never pays a per-env Python call.
//
// These live apart from batched_env.cpp deliberately. The observation
// encoder is large and inlines into write_obs; sharing a translation unit
// with the step loop and the native rollout driver measurably shifted their
// code layout (about 1% of rollout throughput on GCC with LTO) without any
// change to the work being done.
#include "../../include/batched_env.hpp"
#include "../../include/obs.hpp"

#include <cstddef>

namespace catan {

void batched_env_write_actors(const BatchedEnv& env, uint8_t* out) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i)
        out[i] = actor_to_act(env.states[i]);
}

void batched_env_write_status(const BatchedEnv& env, int32_t* out) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        const GameState& s = env.states[i];
        int32_t* row = out + std::size_t(i) * STATUS_INTS;
        row[STATUS_DONE]           = int32_t(s.phase == Phase::ENDED);
        row[STATUS_WINNER]         = int32_t(winner_of(s));
        row[STATUS_TURN_COUNT]     = int32_t(s.turn_count);
        row[STATUS_CURRENT_PLAYER] = int32_t(s.current_player);
    }
}

void batched_env_write_obs(const BatchedEnv& env, float* out) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        write_obs(env.states[i], env.layouts[i],
                  actor_to_act(env.states[i]),
                  out + std::size_t(i) * OBS_SIZE);
    }
}

void batched_env_write_obs_pov(const BatchedEnv& env, const uint8_t* povs,
                               float* out) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        // Masked rather than trusted. The binding checks every seat and
        // raises before it releases the GIL, but `povs` is caller-owned
        // memory that another Python thread could still write to afterwards,
        // and an out-of-range seat here would index past the player arrays.
        write_obs(env.states[i], env.layouts[i],
                  uint8_t(povs[i] & (NUM_PLAYERS - 1)),
                  out + std::size_t(i) * OBS_SIZE);
    }
}

}  // namespace catan

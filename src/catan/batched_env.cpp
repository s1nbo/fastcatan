#include "batched_env.hpp"
#include "rules.hpp"
#include "mask.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

namespace catan {

namespace {

// Stable per-env seed derivation. SplitMix64 of (master, env_id).
inline uint64_t derive_seed(uint64_t master, uint64_t i) noexcept {
    uint64_t x = master ^ (i * 0x9E3779B97F4A7C15ULL);
    return splitmix64(x);
}

inline uint32_t pick_random_legal(const uint64_t mask[MASK_WORDS],
                                  Xoshiro128& rng) noexcept {
    uint32_t count = 0;
    for (uint32_t word = 0; word < MASK_WORDS; ++word)
        count += uint32_t(std::popcount(mask[word]));
    if (count == 0) return NUM_ACTIONS;

    uint32_t target = rng.bounded(count);
    for (uint32_t word = 0; word < MASK_WORDS; ++word) {
        uint64_t bits = mask[word];
        const uint32_t word_count = uint32_t(std::popcount(bits));
        if (target >= word_count) {
            target -= word_count;
            continue;
        }
        while (target > 0) {
            bits &= bits - 1;
            --target;
        }
        return word * 64u + uint32_t(std::countr_zero(bits));
    }
    return NUM_ACTIONS;
}

template <typename T>
T* aligned_array(uint32_t n) noexcept {
    constexpr std::size_t ALIGN = 64;
    std::size_t bytes = sizeof(T) * std::size_t(n);
    bytes = (bytes + ALIGN - 1) & ~(ALIGN - 1);
    void* p = std::aligned_alloc(ALIGN, bytes);
    return reinterpret_cast<T*>(p);
}

inline uint8_t winner_of(const GameState& state) noexcept {
    for (uint8_t player = 0; player < NUM_PLAYERS; ++player) {
        if (state.player_vp[player] >= WIN_VP) return player;
    }
    return NO_PLAYER;
}

}  // namespace

bool batched_env_init(BatchedEnv& env, uint32_t n_envs,
                      uint64_t master_seed) noexcept {
    if (n_envs == 0
        || n_envs > uint32_t(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    env.n            = n_envs;
    env.states       = aligned_array<GameState>(n_envs);
    env.layouts      = aligned_array<BoardLayout>(n_envs);
    env.last_winner  = aligned_array<uint8_t>(n_envs);
    env.seed_counter = master_seed;

    if (!env.states || !env.layouts || !env.last_winner) {
        batched_env_destroy(env);
        return false;
    }

    for (uint32_t i = 0; i < n_envs; ++i) {
        new (&env.states[i])  GameState{};
        new (&env.layouts[i]) BoardLayout{};
        env.last_winner[i] = NO_PLAYER;
    }
    return true;
}

void batched_env_destroy(BatchedEnv& env) noexcept {
    if (env.states)      std::free(env.states);
    if (env.layouts)     std::free(env.layouts);
    if (env.last_winner) std::free(env.last_winner);
    env.states      = nullptr;
    env.layouts     = nullptr;
    env.last_winner = nullptr;
    env.n           = 0;
}

void batched_env_reset(BatchedEnv& env) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static) 
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        uint64_t seed = derive_seed(env.seed_counter, uint32_t(i));
        reset_one(env.states[i], env.layouts[i], seed);
        env.last_winner[i] = NO_PLAYER;
    }
    env.seed_counter += env.n;
}

void batched_env_step(BatchedEnv& env,
                      const uint32_t* actions,
                      uint8_t* dones_out) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        bool done = actions[i] == NO_ACTION
            ? env.states[i].phase == Phase::ENDED
            : step_one(env.states[i], env.layouts[i], actions[i]);
        dones_out[i] = uint8_t(done);

        if (done) env.last_winner[i] = winner_of(env.states[i]);
    }
}

void batched_env_step_autoreset(BatchedEnv& env,
                                const uint32_t* actions,
                                uint8_t* dones_out) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        bool done = actions[i] == NO_ACTION
            ? env.states[i].phase == Phase::ENDED
            : step_one(env.states[i], env.layouts[i], actions[i]);
        dones_out[i] = uint8_t(done);

        if (done) {
            env.last_winner[i] = winner_of(env.states[i]);

            uint64_t seed = derive_seed(env.seed_counter, uint32_t(i));
            reset_one(env.states[i], env.layouts[i], seed);
        }
    }
    env.seed_counter += env.n;
}

uint64_t batched_env_run_random_games(BatchedEnv& env,
                                      uint64_t num_games,
                                      uint64_t& total_steps) noexcept {
    const uint32_t active_slots = uint32_t(std::min<uint64_t>(env.n, num_games));
    const uint64_t seed_base = env.seed_counter;
    uint64_t completed = 0;
    total_steps = 0;

#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:completed,total_steps)
#endif
    for (int32_t slot = 0; slot < int32_t(active_slots); ++slot) {
        GameState& state = env.states[slot];
        BoardLayout& layout = env.layouts[slot];
        const uint64_t games_for_slot =
            (num_games - 1 - uint32_t(slot)) / active_slots + 1;
        for (uint64_t index = 0; index < games_for_slot; ++index) {
            const uint64_t game = uint32_t(slot) + index * active_slots;
            const uint64_t seed = derive_seed(seed_base, game);
            reset_one(state, layout, seed);

            Xoshiro128 picker;
            xoshiro_seed(picker, seed ^ 0xD1B54A32D192ED03ULL);

            bool done = false;
            while (!done) {
                const uint32_t action = pick_random_legal(
                    state.action_mask, picker);
                if (action == NUM_ACTIONS) break;
                done = step_one(state, layout, action);
                ++total_steps;
            }

            if (done) {
                env.last_winner[slot] = winner_of(state);
                ++completed;
            }
        }
    }

    env.seed_counter += num_games;
    return completed;
}

void batched_env_write_masks(const BatchedEnv& env, uint64_t* out) noexcept {
    // Read from the incrementally-maintained s.action_mask field. step_one
    // and reset_one keep it current. ~free vs the previous full recompute.
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        std::memcpy(out + std::size_t(i) * MASK_WORDS,
                    env.states[i].action_mask,
                    sizeof(uint64_t) * MASK_WORDS);
    }
}

void batched_env_save(const BatchedEnv& env, uint8_t* out) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        uint8_t* row = out + std::size_t(i) * SNAPSHOT_BYTES;
        std::memcpy(row, &env.states[i], sizeof(GameState));
        std::memcpy(row + sizeof(GameState), &env.layouts[i],
                    sizeof(BoardLayout));
    }
}

void batched_env_load(BatchedEnv& env, const uint8_t* data) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        const uint8_t* row = data + std::size_t(i) * SNAPSHOT_BYTES;
        std::memcpy(&env.states[i], row, sizeof(GameState));
        std::memcpy(&env.layouts[i], row + sizeof(GameState),
                    sizeof(BoardLayout));
        compute_mask(env.states[i], env.layouts[i],
                     env.states[i].action_mask);
        env.last_winner[i] = env.states[i].phase == Phase::ENDED
            ? winner_of(env.states[i])
            : NO_PLAYER;
    }
}

void batched_env_reseed(BatchedEnv& env, const uint64_t* seeds) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i)
        xoshiro_seed(env.states[i].rng, seeds[i]);
}

}  // namespace catan

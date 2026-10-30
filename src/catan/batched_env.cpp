#include "../../include/batched_env.hpp"
#include "../../include/event_log.hpp"
#include "../../include/mask.hpp"
#include "../../include/rules.hpp"

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

// ---------------------------------------------------------------------
// Event recording. All of it is compiled behind the `Log` template flag of
// step_impl, so a batch with logging disabled runs the original loop body.
// ---------------------------------------------------------------------

// The fields an event row needs from the state before the action runs. A
// trade bundle only exists until TRADE_CONFIRM clears it, so the context has
// to be read ahead of step_one rather than reconstructed afterwards.
struct PreStep {
    uint32_t step_in_game;
    uint16_t turn;
    uint8_t  actor;
    uint8_t  current_player;
    uint8_t  proposer;
    uint8_t  give[NUM_RESOURCES];
    uint8_t  want[NUM_RESOURCES];
    uint8_t  public_vp[NUM_PLAYERS];
    uint8_t  true_vp[NUM_PLAYERS];
    uint8_t  handsize[NUM_PLAYERS];
    bool     was_ended;
    bool     legal;
};

// trade_response value for a seat that accepted (see state.hpp: 0 pending,
// 1 accept, 2 decline, 3 the proposer's own slot).
constexpr uint8_t TRADE_RESPONSE_ACCEPT = 1;

// True when `action` is one of the player-trade actions whose effect
// LOG_TRADES records. `legal` decides whether it took effect at all.
inline bool is_trade_event(uint32_t action, bool legal) noexcept {
    return legal && action >= action::TRADE_OPEN
        && action <= action::TRADE_CANCEL;
}

// A pre-step row is only written for an action the mask selects, so the rest
// of the context is read lazily. Logging trades without logging every action
// then costs four reads per ordinary build or roll instead of thirty.
inline bool wants_context(uint32_t mask, uint32_t action, bool legal) noexcept {
    return (mask & LOG_ACTIONS) || ((mask & LOG_TRADES)
                                    && is_trade_event(action, legal));
}

inline void capture_pre(PreStep& pre, const GameState& s, uint32_t action,
                        uint32_t step_in_game, uint32_t mask) noexcept {
    pre.step_in_game = step_in_game;
    pre.actor        = actor_to_act(s);
    pre.was_ended    = s.phase == Phase::ENDED;
    // step_one applies an action only if its mask bit is set, so this is
    // exactly the condition under which the action has an effect.
    pre.legal = action < NUM_ACTIONS
        && (s.action_mask[action >> 6] & (uint64_t(1) << (action & 63))) != 0;
    if (!wants_context(mask, action, pre.legal)) return;

    pre.turn           = s.turn_count;
    pre.current_player = s.current_player;
    pre.proposer       = s.trade_proposer;
    for (uint32_t r = 0; r < NUM_RESOURCES; ++r) {
        pre.give[r] = s.trade_give[r];
        pre.want[r] = s.trade_want[r];
    }
    for (uint32_t p = 0; p < NUM_PLAYERS; ++p) {
        pre.public_vp[p] = s.player_vp_without_dev[p];
        pre.true_vp[p]   = s.player_vp[p];
        pre.handsize[p]  = s.player_handsize[p];
    }
}

// Every row starts from the same shape: scalars filled, optional fields at
// their "unused" sentinel.
inline void fill_row(int32_t* row, uint32_t env_idx, uint32_t game_id,
                     int32_t type, int32_t action, int32_t turn,
                     int32_t step_in_game, int32_t actor,
                     const uint8_t* public_vp, const uint8_t* true_vp,
                     const uint8_t* handsize) noexcept {
    row[EV_ENV_IDX]      = int32_t(env_idx);
    row[EV_GAME_ID]      = int32_t(game_id);
    row[EV_STEP_IN_GAME] = step_in_game;
    row[EV_TURN]         = turn;
    row[EV_EVENT_TYPE]   = type;
    row[EV_ACTOR]        = actor;
    row[EV_ACTION]       = action;
    row[EV_PROPOSER]     = -1;
    row[EV_PARTNER]      = -1;
    for (uint32_t r = 0; r < NUM_RESOURCES; ++r) {
        row[EV_GIVE + r] = 0;
        row[EV_WANT + r] = 0;
    }
    for (uint32_t p = 0; p < NUM_PLAYERS; ++p) {
        row[EV_PUBLIC_VP + p] = int32_t(public_vp[p]);
        row[EV_TRUE_VP + p]   = int32_t(true_vp[p]);
        row[EV_HANDSIZE + p]  = int32_t(handsize[p]);
    }
    row[EV_WINNER]  = -1;
    row[EV_TURNS]   = -1;
    row[EV_STEPS]   = -1;
    row[EV_SEED_LO] = -1;
    row[EV_SEED_HI] = -1;
}

inline void fill_pre_row(int32_t* row, const BatchedEnv& env, uint32_t i,
                         const PreStep& pre, int32_t type,
                         int32_t action) noexcept {
    fill_row(row, i, env.game_id[i], type, action, int32_t(pre.turn),
             int32_t(pre.step_in_game), int32_t(pre.actor),
             pre.public_vp, pre.true_vp, pre.handsize);
}

// Rows for one stepped action: the action itself, then the player-trade event
// it resolved, if any.
inline void log_step(BatchedEnv& env, uint32_t i, const PreStep& pre,
                     uint32_t action) noexcept {
    const uint32_t mask = env.log.mask;

    if (mask & LOG_ACTIONS) {
        if (int32_t* row = event_log_row(env.log, i))
            fill_pre_row(row, env, i, pre, EVENT_ACTION, int32_t(action));
    }

    if (!(mask & LOG_TRADES) || !is_trade_event(action, pre.legal)) return;

    int32_t type;
    int32_t partner = -1;
    if (action == action::TRADE_OPEN) {
        type = EVENT_TRADE_OPEN;
    } else if (action == action::TRADE_CANCEL) {
        type = EVENT_TRADE_CANCEL;
    } else if (action >= action::TRADE_CONFIRM_BASE) {
        type = EVENT_TRADE_CONFIRM;
        partner = int32_t(action - action::TRADE_CONFIRM_BASE);
    } else {
        // ACCEPT / DECLINE. The simulator coerces an accept into a decline
        // when the responder cannot cover the requested bundle, so the type
        // comes from the response it actually registered; EV_ACTION keeps the
        // action the seat asked for.
        const uint8_t registered =
            uint8_t((env.states[i].trade_response >> (2 * pre.actor)) & 0x3);
        type = registered == TRADE_RESPONSE_ACCEPT ? EVENT_TRADE_ACCEPT
                                                   : EVENT_TRADE_DECLINE;
    }

    if (int32_t* row = event_log_row(env.log, i)) {
        fill_pre_row(row, env, i, pre, type, int32_t(action));
        // The seat that owns the proposal: trade_proposer once a proposal is
        // open, otherwise the turn owner composing the draft.
        row[EV_PROPOSER] = int32_t(
            pre.proposer != NO_PLAYER ? pre.proposer : pre.current_player);
        row[EV_PARTNER] = partner;
        for (uint32_t r = 0; r < NUM_RESOURCES; ++r) {
            row[EV_GIVE + r] = int32_t(pre.give[r]);
            row[EV_WANT + r] = int32_t(pre.want[r]);
        }
    }
}

// A finished episode reports its own final state, not the pre-step one: the
// action that wins the game is also the one that puts the winner on 10 VP.
inline void log_game_end(BatchedEnv& env, uint32_t i, const PreStep& pre,
                         uint32_t action) noexcept {
    int32_t* row = event_log_row(env.log, i);
    if (!row) return;
    const GameState& s = env.states[i];
    fill_row(row, i, env.game_id[i], EVENT_GAME_END, int32_t(action),
             int32_t(s.turn_count), int32_t(pre.step_in_game),
             int32_t(pre.actor), s.player_vp_without_dev, s.player_vp,
             s.player_handsize);
    row[EV_WINNER] = int32_t(winner_of(s));
    row[EV_TURNS]  = int32_t(s.turn_count);
    row[EV_STEPS]  = int32_t(pre.step_in_game + 1);
}

inline void log_game_start(BatchedEnv& env, uint32_t i) noexcept {
    int32_t* row = event_log_row(env.log, i);
    if (!row) return;
    const GameState& s = env.states[i];
    fill_row(row, i, env.game_id[i], EVENT_GAME_START, -1,
             int32_t(s.turn_count), 0, int32_t(actor_to_act(s)),
             s.player_vp_without_dev, s.player_vp, s.player_handsize);
    const uint64_t seed = env.game_seed[i];
    row[EV_SEED_LO] = int32_t(uint32_t(seed & 0xFFFFFFFFu));
    row[EV_SEED_HI] = int32_t(uint32_t(seed >> 32));
}

// Shared body of step and step_autoreset. `Log` keeps every recording branch
// out of the loop when the log is disabled.
template <bool Log, bool Autoreset>
void step_impl(BatchedEnv& env, const uint32_t* actions,
               uint8_t* dones_out) noexcept {
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        const uint32_t action = actions[i];
        const bool stepped = action != NO_ACTION;

        PreStep pre{};
        if constexpr (Log) {
            if (stepped)
                capture_pre(pre, env.states[i], action, env.step_in_game[i],
                            env.log.mask);
        }

        const bool done = stepped
            ? step_one(env.states[i], env.layouts[i], action)
            : env.states[i].phase == Phase::ENDED;
        if (stepped) ++env.step_in_game[i];
        dones_out[i] = uint8_t(done);

        if constexpr (Log) {
            if (stepped) log_step(env, uint32_t(i), pre, action);
        }

        if (done) {
            env.last_winner[i] = winner_of(env.states[i]);

            if constexpr (Log) {
                // Only the transition into a terminal state ends an episode;
                // stepping an already-finished slot does not end it again.
                if (stepped && !pre.was_ended && (env.log.mask & LOG_GAMES))
                    log_game_end(env, uint32_t(i), pre, action);
            }

            if constexpr (Autoreset) {
                ++env.game_id[i];
                env.step_in_game[i] = 0;
                const uint64_t seed = derive_seed(env.seed_counter, uint32_t(i));
                env.game_seed[i] = seed;
                reset_one(env.states[i], env.layouts[i], seed);
                if constexpr (Log) {
                    if (env.log.mask & LOG_GAMES)
                        log_game_start(env, uint32_t(i));
                }
            }
        }
    }
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
    env.game_id      = aligned_array<uint32_t>(n_envs);
    env.step_in_game = aligned_array<uint32_t>(n_envs);
    env.game_seed    = aligned_array<uint64_t>(n_envs);
    env.seed_counter = master_seed;
    env.log          = EventLog{};

    if (!env.states || !env.layouts || !env.last_winner || !env.game_id
        || !env.step_in_game || !env.game_seed) {
        batched_env_destroy(env);
        return false;
    }

    for (uint32_t i = 0; i < n_envs; ++i) {
        new (&env.states[i])  GameState{};
        new (&env.layouts[i]) BoardLayout{};
        env.last_winner[i]  = NO_PLAYER;
        env.game_id[i]      = 0;
        env.step_in_game[i] = 0;
        env.game_seed[i]    = 0;
    }
    return true;
}

void batched_env_destroy(BatchedEnv& env) noexcept {
    event_log_disable(env.log);
    std::free(env.states);
    std::free(env.layouts);
    std::free(env.last_winner);
    std::free(env.game_id);
    std::free(env.step_in_game);
    std::free(env.game_seed);
    env.states       = nullptr;
    env.layouts      = nullptr;
    env.last_winner  = nullptr;
    env.game_id      = nullptr;
    env.step_in_game = nullptr;
    env.game_seed    = nullptr;
    env.n            = 0;
}

void batched_env_reset(BatchedEnv& env) noexcept {
    const bool log_games = (env.log.mask & LOG_GAMES) != 0;
#ifdef FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static) 
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        uint64_t seed = derive_seed(env.seed_counter, uint32_t(i));
        reset_one(env.states[i], env.layouts[i], seed);
        env.last_winner[i]  = NO_PLAYER;
        env.game_id[i]      = 0;
        env.step_in_game[i] = 0;
        env.game_seed[i]    = seed;
        if (log_games) log_game_start(env, uint32_t(i));
    }
    env.seed_counter += env.n;
}

void batched_env_step(BatchedEnv& env,
                      const uint32_t* actions,
                      uint8_t* dones_out) noexcept {
    if (env.log.mask) step_impl<true, false>(env, actions, dones_out);
    else              step_impl<false, false>(env, actions, dones_out);
}

void batched_env_step_autoreset(BatchedEnv& env,
                                const uint32_t* actions,
                                uint8_t* dones_out) noexcept {
    if (env.log.mask) step_impl<true, true>(env, actions, dones_out);
    else              step_impl<false, true>(env, actions, dones_out);
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
            uint64_t game_steps = 0;
            while (!done) {
                const uint32_t action = pick_random_legal(
                    state.action_mask, picker);
                if (action == NUM_ACTIONS) break;
                done = step_one(state, layout, action);
                ++game_steps;
            }
            total_steps += game_steps;

            // Episode bookkeeping stays meaningful across a native rollout,
            // which plays whole games without entering the step loop.
            ++env.game_id[slot];
            env.step_in_game[slot] = uint32_t(game_steps);
            env.game_seed[slot] = seed;

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
    // Copy the incrementally maintained s.action_mask field; step_one and
    // reset_one keep it current.
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
        env.last_winner[i] = winner_of(env.states[i]);
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

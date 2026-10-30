#pragma once
#include <cstddef>
#include <cstdint>

namespace catan {

    // Event-record schema version. Bumped whenever the int32 row layout
    // changes.
    inline constexpr uint32_t EVENT_VERSION = 1;

    // ------------------------------------------------------------------
    // Opt-in native event log.
    //
    // Records are fixed-width int32 rows appended to a per-env buffer from
    // inside the batched step loop, so no locks and no Python calls are
    // involved. The log lives in BatchedEnv, never in GameState, and is not
    // part of a snapshot.
    //
    // Field conventions:
    //   -1          the field is unused by this event type
    //   NO_PLAYER   a seat-valued field that names no seat
    // Context fields (turn, actor, proposer, give, want, public_vp, true_vp,
    // handsize, step_in_game) describe the state BEFORE the action ran, which
    // is the only point where a trade bundle is still readable. EVENT_GAME_END
    // is the exception: it reports the finished game, so its context is the
    // post-step state.
    // ------------------------------------------------------------------

    // Row layout. Multi-slot fields list their half-open range.
    inline constexpr uint32_t EV_ENV_IDX      = 0;
    inline constexpr uint32_t EV_GAME_ID      = 1;   // per-env episode counter
    inline constexpr uint32_t EV_STEP_IN_GAME = 2;
    inline constexpr uint32_t EV_TURN         = 3;
    inline constexpr uint32_t EV_EVENT_TYPE   = 4;
    inline constexpr uint32_t EV_ACTOR        = 5;
    inline constexpr uint32_t EV_ACTION       = 6;   // triggering action
    inline constexpr uint32_t EV_PROPOSER     = 7;
    inline constexpr uint32_t EV_PARTNER      = 8;   // trade confirm only
    inline constexpr uint32_t EV_GIVE         = 9;   // [9, 14)
    inline constexpr uint32_t EV_WANT         = 14;  // [14, 19)
    inline constexpr uint32_t EV_PUBLIC_VP    = 19;  // [19, 23) vp_without_dev
    inline constexpr uint32_t EV_TRUE_VP      = 23;  // [23, 27)
    inline constexpr uint32_t EV_HANDSIZE     = 27;  // [27, 31)
    inline constexpr uint32_t EV_WINNER       = 31;  // game end
    inline constexpr uint32_t EV_TURNS        = 32;  // game end
    inline constexpr uint32_t EV_STEPS        = 33;  // game end
    inline constexpr uint32_t EV_SEED_LO      = 34;  // game start
    inline constexpr uint32_t EV_SEED_HI      = 35;  // game start
    inline constexpr uint32_t EVENT_INTS      = 36;

    // Event types.
    inline constexpr int32_t EVENT_GAME_START    = 0;
    inline constexpr int32_t EVENT_GAME_END      = 1;
    inline constexpr int32_t EVENT_ACTION        = 2;
    inline constexpr int32_t EVENT_TRADE_OPEN    = 3;
    inline constexpr int32_t EVENT_TRADE_ACCEPT  = 4;
    inline constexpr int32_t EVENT_TRADE_DECLINE = 5;
    inline constexpr int32_t EVENT_TRADE_CONFIRM = 6;
    inline constexpr int32_t EVENT_TRADE_CANCEL  = 7;

    // Recording mask bits.
    inline constexpr uint32_t LOG_TRADES  = 1u;  // actions 278..285
    inline constexpr uint32_t LOG_GAMES   = 2u;  // game start + game end
    inline constexpr uint32_t LOG_ACTIONS = 4u;  // every (actor, action)
    inline constexpr uint32_t LOG_ALL     = LOG_TRADES | LOG_GAMES | LOG_ACTIONS;

    // Per-env append cursor. One cache line each so two threads appending to
    // neighbouring envs never write the same line.
    struct alignas(64) EventCursor {
        uint32_t count;    // pending rows
        uint32_t dropped;  // rows refused since the log was enabled
        uint32_t pad[14];
    };
    static_assert(sizeof(EventCursor) == 64);

    struct EventLog {
        uint32_t     mask;      // 0 when disabled
        uint32_t     capacity;  // rows per env
        uint32_t     stride;    // int32 per env slab, >= capacity * EVENT_INTS
        int32_t*     rows;      // n * stride
        EventCursor* cursors;   // n
    };

    // Allocate buffers for `n` envs and start recording `mask`. Replaces any
    // previous buffers and clears the pending and dropped counters. Returns
    // false if allocation fails or the request does not fit in memory.
    bool event_log_enable(EventLog& log, uint32_t n, uint32_t mask,
                          uint32_t capacity) noexcept;

    // Stop recording and free buffers. Safe on a zero-initialized log.
    void event_log_disable(EventLog& log) noexcept;

    // Discard pending rows and reset the dropped counter, keeping buffers.
    void event_log_reset(EventLog& log, uint32_t n) noexcept;

    uint64_t event_log_pending(const EventLog& log, uint32_t n) noexcept;
    uint64_t event_log_dropped(const EventLog& log, uint32_t n) noexcept;

    // Copy up to `out_rows` pending rows in (env_idx, append order) into `out`
    // and clear what was copied. An env that does not fit entirely keeps its
    // remaining rows, compacted to the front of its slab, so repeated drains
    // never lose a record. Returns the number of rows written.
    uint32_t event_log_drain(EventLog& log, uint32_t n, int32_t* out,
                             uint32_t out_rows) noexcept;

    // Reserve one row for env `env_idx`, or return nullptr when that env's
    // buffer is full (which counts a drop). Hot path: called from inside the
    // parallel step loop, touching only this env's own slab and cursor.
    inline int32_t* event_log_row(EventLog& log, uint32_t env_idx) noexcept {
        EventCursor& cursor = log.cursors[env_idx];
        if (cursor.count >= log.capacity) {
            ++cursor.dropped;
            return nullptr;
        }
        int32_t* row = log.rows
            + std::size_t(env_idx) * log.stride
            + std::size_t(cursor.count) * EVENT_INTS;
        ++cursor.count;
        return row;
    }

}  // namespace catan

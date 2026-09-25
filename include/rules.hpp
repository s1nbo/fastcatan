#pragma once
#include <cstdint>
#include "state.hpp"

namespace catan {

    // Initialize one episode. Randomizes BoardLayout
    void reset_one(GameState& s, BoardLayout& b, uint64_t seed) noexcept;

    // Advance one game by one action and return whether it is terminal.
    // Mask-illegal actions are strict no-ops.
    bool step_one(GameState& s, const BoardLayout& b,
                  uint32_t action) noexcept;

    // Maximum number of physical outcomes for one action: the eleven 2d6
    // totals. Deterministic actions produce one outcome.
    inline constexpr uint32_t MAX_ACTION_OUTCOMES = 11;

    // Enumerate the exact simulator transitions for one action without
    // sampling chance. Dice rolls, development-card draws, and robber steals
    // fan out according to the physical game state. Illegal actions produce
    // one unchanged outcome with probability 1. The caller provides arrays of
    // at least MAX_ACTION_OUTCOMES elements. Each child consumes the same RNG
    // draw as a sampled transition, while the source remains unchanged.
    // Returns the number written.
    uint32_t enumerate_action_outcomes(
        const GameState& s, const BoardLayout& b, uint32_t action,
        GameState* out_states, double* out_probabilities) noexcept;

    // Action ID layout
    namespace action {
        inline constexpr uint32_t SETTLE_BASE      = 0;    // [0, 54)
        inline constexpr uint32_t CITY_BASE        = 54;   // [54, 108)
        inline constexpr uint32_t ROAD_BASE        = 108;  // [108, 180)
        inline constexpr uint32_t ROLL_DICE        = 180;
        inline constexpr uint32_t END_TURN         = 181;
        inline constexpr uint32_t DISCARD_BASE     = 182;  // [182, 187) discard 1 of resource r
        inline constexpr uint32_t MOVE_ROBBER_BASE = 187;  // [187, 206) move robber to hex h
        inline constexpr uint32_t STEAL_BASE       = 206;  // [206, 210) steal from player p
        inline constexpr uint32_t TRADE_BASE       = 210;  // [210, 235) bank/port trade: give*5 + get
        inline constexpr uint32_t BUY_DEV          = 235;
        inline constexpr uint32_t PLAY_KNIGHT          = 236;
        inline constexpr uint32_t PLAY_ROAD_BUILDING   = 237;
        inline constexpr uint32_t PLAY_YEAR_OF_PLENTY  = 238;  // [238, 263) base + give1*5 + give2 (25 options)
        inline constexpr uint32_t PLAY_MONOPOLY        = 263;  // [263, 268) base + resource

        // Player-to-player trade (compositional). Compose only valid post-roll
        // with no active flag; OPEN/ACCEPT/DECLINE/CONFIRM/CANCEL drive the flow.
        inline constexpr uint32_t TRADE_ADD_GIVE_BASE    = 268;  // [268, 273) +1 of give resource r
        inline constexpr uint32_t TRADE_ADD_WANT_BASE    = 273;  // [273, 278) +1 of want resource r
        inline constexpr uint32_t TRADE_OPEN             = 278;  // finalize proposal, broadcast to opponents
        inline constexpr uint32_t TRADE_ACCEPT           = 279;  // responder accepts the open proposal
        inline constexpr uint32_t TRADE_DECLINE          = 280;  // responder declines
        inline constexpr uint32_t TRADE_CONFIRM_BASE     = 281;  // [281, 285) proposer picks accepting partner p
        inline constexpr uint32_t TRADE_CANCEL           = 285;  // proposer cancels (compose or post-response)
    }

}  // namespace catan

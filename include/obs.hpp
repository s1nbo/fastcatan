#pragma once
#include <cstdint>
#include "state.hpp"
#include "topology.hpp"

namespace catan {

    // Observation schema version. Bumped whenever the float layout changes;
    // RL agents trained against an older version must retrain.
    inline constexpr uint32_t OBS_VERSION = 1;

    // ------------------------------------------------------------------
    // Point-of-view observation encoder.
    //
    // The encoding is POV-relative: the pov seat always occupies relative
    // slot 0 and opponents follow clockwise as +1, +2, +3. It carries public
    // information plus the pov's own private information and nothing else.
    // In particular it exposes the development deck's SIZE but never its
    // composition (the per-type remainder would reveal which card types
    // opponents hold), and the player-to-player trade draft only while it is
    // public (an open proposal) or owned by the pov.
    // ------------------------------------------------------------------

    // Per-player block, in relative-seat order:
    //  +0      victory points      (pov: total; opponents: public only)
    //  +1      hand size
    //  +2      development cards held
    //  +3      knights played
    //  +4      longest road length
    //  +5..+7  settlements / cities / roads remaining
    //  +8..+13 port access bits 0..5
    //  +14     cards still owed to a discard
    //  +15     is the turn owner
    //  +16     is the seat that must act now
    inline constexpr uint32_t OBS_PER_PLAYER = 17;

    // The pov's private hand: resources(5), development cards by type(5),
    // development cards bought this turn(5).
    inline constexpr uint32_t OBS_SELF_PRIVATE = 15;

    // Section offsets into the flat float vector.
    inline constexpr uint32_t OBS_OFF_PLAYERS      = 0;
    inline constexpr uint32_t OBS_OFF_SELF         = OBS_OFF_PLAYERS + NUM_PLAYERS * OBS_PER_PLAYER;
    inline constexpr uint32_t OBS_OFF_NODES        = OBS_OFF_SELF + OBS_SELF_PRIVATE;
    inline constexpr uint32_t OBS_OFF_EDGES        = OBS_OFF_NODES + 2 * NUM_PLAYERS * topology::NUM_NODES;
    inline constexpr uint32_t OBS_OFF_HEX_RESOURCE = OBS_OFF_EDGES + NUM_PLAYERS * topology::NUM_EDGES;
    inline constexpr uint32_t OBS_OFF_HEX_NUMBER   = OBS_OFF_HEX_RESOURCE + 6 * topology::NUM_HEXES;
    inline constexpr uint32_t OBS_OFF_PORT_TYPE    = OBS_OFF_HEX_NUMBER + topology::NUM_HEXES;
    inline constexpr uint32_t OBS_OFF_ROBBER       = OBS_OFF_PORT_TYPE + 6 * topology::NUM_PORTS;
    inline constexpr uint32_t OBS_OFF_PHASE        = OBS_OFF_ROBBER + topology::NUM_HEXES;
    inline constexpr uint32_t OBS_OFF_FLAG         = OBS_OFF_PHASE + 4;
    inline constexpr uint32_t OBS_OFF_DICE         = OBS_OFF_FLAG + 8;
    inline constexpr uint32_t OBS_OFF_TURN         = OBS_OFF_DICE + 13;
    inline constexpr uint32_t OBS_OFF_BANK         = OBS_OFF_TURN + 1;
    inline constexpr uint32_t OBS_OFF_DEV_DECK     = OBS_OFF_BANK + NUM_RESOURCES;
    inline constexpr uint32_t OBS_OFF_LONGEST_ROAD = OBS_OFF_DEV_DECK + 1;
    inline constexpr uint32_t OBS_OFF_LARGEST_ARMY = OBS_OFF_LONGEST_ROAD + 5;
    inline constexpr uint32_t OBS_OFF_START_PLAYER = OBS_OFF_LARGEST_ARMY + 5;
    inline constexpr uint32_t OBS_OFF_FREE_ROADS   = OBS_OFF_START_PLAYER + 4;
    inline constexpr uint32_t OBS_OFF_DEV_PLAYED   = OBS_OFF_FREE_ROADS + 1;
    inline constexpr uint32_t OBS_OFF_COMPOSE      = OBS_OFF_DEV_PLAYED + 1;
    inline constexpr uint32_t OBS_OFF_TRADE_PROP   = OBS_OFF_COMPOSE + 1;
    inline constexpr uint32_t OBS_OFF_TRADE_GIVE   = OBS_OFF_TRADE_PROP + 5;
    inline constexpr uint32_t OBS_OFF_TRADE_WANT   = OBS_OFF_TRADE_GIVE + NUM_RESOURCES;
    inline constexpr uint32_t OBS_OFF_TRADE_RESP   = OBS_OFF_TRADE_WANT + NUM_RESOURCES;
    inline constexpr uint32_t OBS_SIZE             = OBS_OFF_TRADE_RESP + 4 * NUM_PLAYERS;

    static_assert(OBS_SIZE == 1089,
                  "OBS_SIZE changed; bump OBS_VERSION and update README/tests");
    static_assert((NUM_PLAYERS & (NUM_PLAYERS - 1)) == 0,
                  "relative-seat arithmetic and pov masking assume a power of two");

    // Encode `s` from `pov`'s perspective into `out[0..OBS_SIZE)`. Counts are
    // normalized by structural Catan maxima (documented in README); booleans
    // and one-hot slots are 0.0 / 1.0. Occasional values exceed 1 (a hand can
    // spike past 25 after a monopoly); the goal is comparable scale, not a
    // hard bound.
    void write_obs(const GameState& s, const BoardLayout& b,
                   uint8_t pov, float* out) noexcept;

}  // namespace catan

// Observation encoder: GameState -> flat float tensor from one seat's POV.
//
// Section offsets live in obs.hpp (OBS_OFF_*); the normalization divisors
// live in namespace norm below and are mirrored in the README table. Every
// field is either public information or the pov's own private information --
// see the leakage test in tests/test_obs.py, which randomizes the opponents'
// hidden holdings and requires a byte-identical observation.
#include "obs.hpp"
#include "topology.hpp"

#include <cassert>

namespace catan {

namespace {

// Structural Catan maxima, baked into the frozen observation so consumers see
// roughly [0,1] inputs. Changing any of these changes the encoding and
// requires an OBS_VERSION bump.
namespace norm {
constexpr float VP        = 10.0f;  // win at 10
constexpr float HAND      = 25.0f;  // a hand can spike past 20 on a monopoly
constexpr float DEV       = 10.0f;  // development cards held / bought
constexpr float KNIGHTS   = 10.0f;
constexpr float ROADLEN   = 15.0f;  // 15 road pieces
constexpr float SETTLE    = 5.0f;   // settlements available (starts at 5)
constexpr float CITY      = 4.0f;   // cities available (starts at 4)
constexpr float ROAD      = 15.0f;  // roads available (starts at 15)
constexpr float DISCARD   = 10.0f;  // cards owed on a 7
constexpr float RES       = 19.0f;  // per-resource supply
constexpr float BANK      = 19.0f;
constexpr float DEVDECK   = 25.0f;  // full development deck
constexpr float HEXNUM    = 12.0f;  // highest dice total on a hex
constexpr float TURN      = 400.0f; // typical game length, not a hard cap
constexpr float FREEROADS = 2.0f;   // road building grants up to 2
constexpr float TRADE     = 19.0f;  // trade bundle counts (resource-bounded)
constexpr float COMPOSE    = float(MAX_TRADE_COMPOSE_PER_TURN);
}  // namespace norm

// Seat `player` expressed relative to `self`: 0 = self, 1..3 = clockwise.
inline uint8_t relseat(uint8_t self, uint8_t player) noexcept {
    return uint8_t((player + NUM_PLAYERS - self) & 0x3);
}

struct Writer {
    float* p;
    inline void put(float v) noexcept { *p++ = v; }
    inline void zero(uint32_t n) noexcept { for (uint32_t i = 0; i < n; ++i) *p++ = 0.0f; }
    // One-hot of size `n`. idx in [0, n) sets that slot; idx >= n (a sentinel)
    // leaves every slot at zero.
    inline void onehot(uint32_t idx, uint32_t n) noexcept {
        for (uint32_t i = 0; i < n; ++i) *p++ = (i == idx) ? 1.0f : 0.0f;
    }
    // Seat one-hot over [self, +1, +2, +3, none].
    inline void seat(uint8_t self, uint8_t player) noexcept {
        onehot(player == NO_PLAYER ? 4u : relseat(self, player), 5u);
    }
};

// OBS_PER_PLAYER floats. `is_self` selects the pov's total victory points over
// the publicly visible count, which is the only private field in this block.
inline void encode_player(Writer& w, const GameState& s, uint8_t pl,
                          uint8_t actor, bool is_self) noexcept {
    w.put(float(is_self ? s.player_vp[pl] : s.player_vp_without_dev[pl]) / norm::VP);
    w.put(float(s.player_handsize[pl])         / norm::HAND);
    w.put(float(s.player_total_dev[pl])        / norm::DEV);
    w.put(float(s.player_knights_played[pl])   / norm::KNIGHTS);
    w.put(float(s.player_road_length[pl])      / norm::ROADLEN);
    w.put(float(s.player_settlement_count[pl]) / norm::SETTLE);
    w.put(float(s.player_city_count[pl])       / norm::CITY);
    w.put(float(s.player_road_count[pl])       / norm::ROAD);
    const uint8_t ports = s.player_ports[pl];
    for (uint8_t bit = 0; bit < 6; ++bit) w.put(float((ports >> bit) & 1));
    w.put(float(s.player_discard_remaining[pl]) / norm::DISCARD);
    w.put(float(s.current_player == pl ? 1 : 0));
    w.put(float(actor == pl ? 1 : 0));
}

}  // namespace

void write_obs(const GameState& s, const BoardLayout& b,
               uint8_t pov, float* out) noexcept {
    Writer w{out};
    const uint8_t actor = actor_to_act(s);

    // ----- Per-player blocks, relative-seat order: self, +1, +2, +3 -----
    for (uint8_t rel = 0; rel < NUM_PLAYERS; ++rel) {
        const uint8_t pl = uint8_t((pov + rel) & 0x3);
        encode_player(w, s, pl, actor, /*is_self=*/rel == 0);
    }

    // ----- The pov's own hand -----
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r)
        w.put(float(s.player_resources[pov][r]) / norm::RES);
    for (uint8_t d = 0; d < 5; ++d)
        w.put(float(s.player_dev[pov][d]) / norm::DEV);
    for (uint8_t d = 0; d < 5; ++d)
        w.put(float(s.player_dev_bought_this_turn[pov][d]) / norm::DEV);
    assert(w.p == out + OBS_OFF_NODES);

    // ----- Nodes: 2 channels per relative seat, {settlement, city} -----
    for (uint8_t n = 0; n < topology::NUM_NODES; ++n) {
        const uint8_t packed = s.node[n];
        const uint8_t level = node_level(packed);
        if (level == NODE_EMPTY) {
            w.zero(2u * NUM_PLAYERS);
            continue;
        }
        const uint8_t rel = relseat(pov, node_owner(packed));
        for (uint8_t r = 0; r < NUM_PLAYERS; ++r) {
            w.put((r == rel && level == NODE_SETTLEMENT) ? 1.0f : 0.0f);
            w.put((r == rel && level == NODE_CITY)       ? 1.0f : 0.0f);
        }
    }

    // ----- Edges: one road channel per relative seat -----
    for (uint8_t e = 0; e < topology::NUM_EDGES; ++e) {
        const uint8_t owner = s.edge[e];
        if (owner == NO_PLAYER) {
            w.zero(NUM_PLAYERS);
            continue;
        }
        w.onehot(relseat(pov, owner), NUM_PLAYERS);
    }

    // ----- Static board -----
    for (uint8_t h = 0; h < topology::NUM_HEXES; ++h) w.onehot(b.hex_resource[h], 6u);
    for (uint8_t h = 0; h < topology::NUM_HEXES; ++h)
        w.put(float(b.hex_number[h]) / norm::HEXNUM);
    for (uint8_t p = 0; p < topology::NUM_PORTS; ++p) w.onehot(b.port_type[p], 6u);
    w.onehot(s.robber_hex, topology::NUM_HEXES);
    assert(w.p == out + OBS_OFF_PHASE);

    // ----- Game state -----
    w.onehot(uint32_t(s.phase), 4u);
    w.onehot(uint32_t(s.flag), 8u);
    w.onehot(s.dice_roll, 13u);  // slot 0 = not yet rolled; slot 1 unused
    w.put(float(s.turn_count) / norm::TURN);
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) w.put(float(s.bank[r]) / norm::BANK);

    // Deck SIZE only. The per-type remainder is not public information: it
    // would reveal which development-card types the opponents are holding.
    uint32_t deck_size = 0;
    for (uint8_t d = 0; d < 5; ++d) deck_size += s.dev_deck[d];
    w.put(float(deck_size) / norm::DEVDECK);

    w.seat(pov, s.longest_road_owner);
    w.seat(pov, s.largest_army_owner);
    w.onehot(relseat(pov, s.start_player), 4u);
    w.put(float(s.free_roads_remaining) / norm::FREEROADS);
    w.put(float(s.dev_card_played ? 1 : 0));

    // The per-turn compose budget belongs to the turn owner alone.
    w.put(pov == s.current_player
              ? float(s.trade_compose_count) / norm::COMPOSE
              : 0.0f);
    assert(w.p == out + OBS_OFF_TRADE_PROP);

    // ----- Player-to-player trade -----
    // An unopened draft is private to the player composing it, so the bundles
    // appear only once the proposal is public or when the pov owns the draft.
    const bool trade_open = s.trade_proposer != NO_PLAYER;
    const bool bundles_visible = trade_open || pov == s.current_player;
    w.seat(pov, s.trade_proposer);
    if (bundles_visible) {
        for (uint8_t r = 0; r < NUM_RESOURCES; ++r)
            w.put(float(s.trade_give[r]) / norm::TRADE);
        for (uint8_t r = 0; r < NUM_RESOURCES; ++r)
            w.put(float(s.trade_want[r]) / norm::TRADE);
    } else {
        w.zero(2u * NUM_RESOURCES);
    }

    // Responses in relative-seat order, 4-slot one-hot
    // [PENDING, ACCEPT, DECLINE, N/A]; all zero while no proposal is open.
    if (trade_open) {
        for (uint8_t rel = 0; rel < NUM_PLAYERS; ++rel) {
            const uint8_t pl = uint8_t((pov + rel) & 0x3);
            w.onehot(uint32_t((s.trade_response >> (2 * pl)) & 0x3), 4u);
        }
    } else {
        w.zero(4u * NUM_PLAYERS);
    }

    assert(w.p == out + OBS_SIZE);
}

}  // namespace catan

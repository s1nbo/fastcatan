// Determinization: resample the information hidden from one seat.
//
// Every deal draws cards one at a time, uniformly without replacement, from a
// multiset of counts. That is exactly a shuffle of the pooled cards followed
// by dealing them out, so each consistent assignment of physical cards to
// slots is equally likely. Opponents are dealt in seat order.
#include "determinize.hpp"
#include "mask.hpp"
#include "rng.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace catan {

namespace {

// Layout indices (match state.hpp / rules.cpp).
constexpr uint8_t NUM_DEV_TYPES = 5;
constexpr uint8_t DEV_VP        = 1;
constexpr uint8_t TR_ACCEPT     = 1;

// Development-card deals that would put an opponent at WIN_VP are redrawn.
// The true assignment always passes unless an opponent legitimately sits at
// WIN_VP between turns (Longest Road can change hands off-turn), so a deal is
// normally accepted within a few attempts. The bound only stops a pathological
// pool that cannot pass; the last deal is then kept as is.
constexpr uint32_t MAX_DEV_DEALS = 1024;

// Remove one card uniformly at random from `pool` (`total` cards over `n`
// kinds) and return its kind. Requires total > 0.
inline uint8_t draw(uint8_t* pool, uint8_t n, uint32_t& total,
                    Xoshiro128& r) noexcept {
    assert(total > 0);
    uint32_t pick = r.bounded(total);
    uint8_t k = 0;
    while (k + 1 < n && pick >= pool[k]) {
        pick -= pool[k];
        ++k;
    }
    --pool[k];
    --total;
    return k;
}

inline uint32_t sum5(const uint8_t v[5]) noexcept {
    return uint32_t(v[0]) + v[1] + v[2] + v[3] + v[4];
}

// Hand `bundle` to `seat` straight from the pool before the uniform deal.
// Clamped so an inconsistent hand-built state cannot underflow anything.
inline void reserve(GameState& s, uint8_t seat, const uint8_t bundle[5],
                    uint8_t pool[5], uint32_t& total, uint8_t need[4]) noexcept {
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        uint8_t take = std::min({bundle[r], pool[r], need[seat]});
        s.player_resources[seat][r] += take;
        pool[r] -= take;
        need[seat] -= take;
        total -= take;
    }
}

// Pool the opponents' resource cards and redeal them, keeping hand sizes.
// Belief-weighted sampling (a per-opponent resource-weight matrix) would
// replace the uniform draw in the final loop.
void deal_resources(GameState& s, uint8_t pov, Xoshiro128& r) noexcept {
    uint8_t pool[NUM_RESOURCES] = {};
    uint8_t need[NUM_PLAYERS] = {};
    uint32_t total = 0;
    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
        if (o == pov) continue;
        for (uint8_t k = 0; k < NUM_RESOURCES; ++k) {
            pool[k] += s.player_resources[o][k];
            s.player_resources[o][k] = 0;
        }
        need[o] = s.player_handsize[o];
        total += need[o];
    }

    // An open proposal makes the offered bundle public, and an ACCEPT proves
    // the responder holds the requested one. Deal those first.
    if (s.flag == Flag::TRADE_PENDING && s.trade_proposer < NUM_PLAYERS) {
        const uint8_t proposer = s.trade_proposer;
        if (proposer != pov)
            reserve(s, proposer, s.trade_give, pool, total, need);
        for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
            if (o == pov || o == proposer) continue;
            if (((s.trade_response >> (2 * o)) & 0x3) == TR_ACCEPT)
                reserve(s, o, s.trade_want, pool, total, need);
        }
    }

    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
        if (o == pov) continue;
        for (uint8_t i = 0; i < need[o] && total > 0; ++i)
            s.player_resources[o][draw(pool, NUM_RESOURCES, total, r)] += 1;
    }
}

// Pool the opponents' unplayed development cards with the deck and redeal,
// keeping each opponent's held and bought-this-turn counts.
void deal_dev_cards(GameState& s, uint8_t pov, Xoshiro128& r) noexcept {
    uint8_t pool[NUM_DEV_TYPES];
    std::memcpy(pool, s.dev_deck, sizeof(pool));
    uint8_t held[NUM_PLAYERS] = {};
    uint8_t pending[NUM_PLAYERS] = {};
    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
        if (o == pov) continue;
        for (uint8_t d = 0; d < NUM_DEV_TYPES; ++d) {
            pool[d] += s.player_dev[o][d] + s.player_dev_bought_this_turn[o][d];
            held[o] += s.player_dev[o][d];
            pending[o] += s.player_dev_bought_this_turn[o][d];
        }
    }
    const uint32_t pool_total = sum5(pool);

    for (uint32_t attempt = 0; attempt < MAX_DEV_DEALS; ++attempt) {
        uint8_t left[NUM_DEV_TYPES];
        std::memcpy(left, pool, sizeof(left));
        uint32_t total = pool_total;

        // Bought-this-turn slots first, from the non-VP cards only: a bought
        // victory-point card goes straight into the held pile (rules.cpp).
        const uint8_t vp_aside = left[DEV_VP];
        left[DEV_VP] = 0;
        total -= vp_aside;
        for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
            if (o == pov) continue;
            std::memset(s.player_dev_bought_this_turn[o], 0, NUM_DEV_TYPES);
            for (uint8_t i = 0; i < pending[o] && total > 0; ++i)
                s.player_dev_bought_this_turn[o][draw(left, NUM_DEV_TYPES, total, r)] += 1;
        }
        left[DEV_VP] = vp_aside;
        total += vp_aside;

        for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
            if (o == pov) continue;
            std::memset(s.player_dev[o], 0, NUM_DEV_TYPES);
            for (uint8_t i = 0; i < held[o] && total > 0; ++i)
                s.player_dev[o][draw(left, NUM_DEV_TYPES, total, r)] += 1;
        }
        std::memcpy(s.dev_deck, left, sizeof(left));

        bool plausible = true;
        for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
            if (o == pov || s.player_vp_without_dev[o] >= WIN_VP) continue;
            if (s.player_vp_without_dev[o] + s.player_dev[o][DEV_VP] >= WIN_VP) {
                plausible = false;
                break;
            }
        }
        if (plausible) break;
    }

    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
        if (o == pov) continue;
        s.player_vp[o] = uint8_t(s.player_vp_without_dev[o] + s.player_dev[o][DEV_VP]);
    }
}

}  // namespace

void determinize(GameState& s, const BoardLayout& b, uint8_t pov,
                 uint64_t seed) noexcept {
    assert(pov < NUM_PLAYERS);
    if (pov >= NUM_PLAYERS) return;

    // Two independent streams: one for the deal, one for the game's future.
    uint64_t key = seed;
    Xoshiro128 deal_rng;
    xoshiro_seed(deal_rng, splitmix64(key));
    xoshiro_seed(s.rng, splitmix64(key));

    if (s.phase != Phase::ENDED) {
        deal_resources(s, pov, deal_rng);
        deal_dev_cards(s, pov, deal_rng);

        // An unopened draft belongs to the turn owner alone.
        if (s.trade_proposer == NO_PLAYER && s.current_player != pov) {
            std::memset(s.trade_give, 0, sizeof(s.trade_give));
            std::memset(s.trade_want, 0, sizeof(s.trade_want));
        }
    }

    compute_mask(s, b, s.action_mask);
}

}  // namespace catan

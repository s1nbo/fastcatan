#pragma once
#include <cstdint>
#include "state.hpp"

namespace catan {

    // Information-set sampling for search (ISMCTS, PIMC, ...).
    //
    // Replaces everything `pov` cannot observe with a uniformly random
    // assignment that is consistent with what `pov` can observe, and reseeds
    // the engine RNG so that future dice, development draws and robber steals
    // are fresh. Without the reseed a copy taken with snapshot() replays the
    // real game's future chance events.
    //
    // Kept: the board, buildings, roads, robber, bank, awards, VP from pieces,
    // knights played, phase / flag / sub-phase bookkeeping, every hand size,
    // every development-card count, and pov's own resources and development
    // cards.
    //
    // Resampled:
    //  * Opponent resources. The opponents' cards are pooled and redealt so
    //    that each keeps their hand size; per-resource totals stay at
    //    19 - bank - pov's hand. Cards an open trade makes public are dealt
    //    first: the proposer keeps the offered bundle and every responder who
    //    accepted keeps the requested bundle (the engine force-declines a
    //    responder who cannot pay, so an ACCEPT proves the cards are held).
    //  * Opponent development cards. The opponents' unplayed cards (held,
    //    bought this turn, victory points) and the deck are pooled and redealt
    //    so that each opponent keeps their held count and their bought-this-
    //    turn count; the remainder becomes the deck. Bought-this-turn slots
    //    never receive a victory-point card, matching the engine. Total VP
    //    follows the dealt victory-point cards. A deal that lifts an opponent
    //    to WIN_VP is redrawn (bounded; see determinize.cpp), except for an
    //    opponent whose public VP is already at WIN_VP.
    //  * An opponent's unopened trade draft is private to them and is cleared.
    //  * The RNG, from `seed`.
    //
    // The result is a pure function of (s, pov, seed) and ends with a mask
    // refresh. A finished game has nothing left to sample: only the RNG is
    // reseeded. Mutates `s` in place, so call it on a copy, never on the
    // state of the game being played.
    void determinize(GameState& s, const BoardLayout& b, uint8_t pov,
                     uint64_t seed) noexcept;

}  // namespace catan

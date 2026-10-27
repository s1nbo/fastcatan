// tests/test_rules.cpp — rule scenarios on hand-built positions.
//
// Each scenario places pieces directly on an empty MAIN-phase board, then
// drives the engine through step_one so the rules run exactly as in play.
// Expected behavior follows the catan.com base-game FAQ.
//
// Board rows are simple node paths: 0-6, 7-15, 16-26, 27-37, 38-46, 47-53.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mask.hpp"
#include "rules.hpp"
#include "state.hpp"
#include "topology.hpp"

using namespace catan;

namespace {

int failures = 0;

void expect(bool ok, const char* scenario, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL [%s] %s\n", scenario, what);
    ++failures;
}

uint8_t edge_between(uint8_t a, uint8_t b) {
    for (uint8_t k = 0; k < topology::MAX_EDGES_PER_NODE; ++k) {
        uint8_t e = topology::node_to_edge[a][k];
        if (e == topology::NO_EDGE) break;
        if (topology::edge_to_node[e][0] == b || topology::edge_to_node[e][1] == b)
            return e;
    }
    std::fprintf(stderr, "test bug: nodes %u and %u are not adjacent\n", a, b);
    std::abort();
}

// Lay `player`'s roads along the row path first..last (consecutive node ids).
void lay_road(GameState& s, uint8_t player, uint8_t first, uint8_t last) {
    for (uint8_t n = first; n < last; ++n)
        s.edge[edge_between(n, uint8_t(n + 1))] = player;
}

// Post-roll MAIN position on an empty board with `turn` to act. Every player
// holds 4 of each resource, leaving 3 of each in the bank.
void empty_main(GameState& s, BoardLayout& b, uint8_t turn) {
    reset_one(s, b, 0);
    std::memset(s.node, NODE_EMPTY, sizeof(s.node));
    std::memset(s.edge, NO_PLAYER, sizeof(s.edge));
    s.phase = Phase::MAIN;
    s.flag = Flag::NONE;
    s.current_player = turn;
    s.dice_roll = 8;
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        for (uint8_t r = 0; r < NUM_RESOURCES; ++r)
            s.player_resources[p][r] = 4;
        s.player_handsize[p] = 4 * NUM_RESOURCES;
    }
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) s.bank[r] = 3;
}

void give_vp(GameState& s, uint8_t player, uint8_t vp) {
    s.player_vp[player] = vp;
    s.player_vp_without_dev[player] = vp;
}

bool legal(const GameState& s, const BoardLayout& b, uint32_t action) {
    uint64_t mask[MASK_WORDS];
    compute_mask(s, b, mask);
    return mask[action >> 6] & (uint64_t(1) << (action & 63));
}

// Apply a legal action; reports and skips it when the mask rejects it.
bool act(GameState& s, const BoardLayout& b, uint32_t action, const char* scenario) {
    if (!legal(s, b, action)) {
        std::fprintf(stderr, "FAIL [%s] action %u is not legal\n", scenario, action);
        ++failures;
        return s.phase == Phase::ENDED;
    }
    return step_one(s, b, action);
}

uint32_t road(uint8_t a, uint8_t b) { return action::ROAD_BASE + edge_between(a, b); }

void longest_road_tie_is_set_aside() {
    const char* name = "longest road tie";
    GameState s; BoardLayout b;
    empty_main(s, b, 3);
    lay_road(s, 0, 7, 13);              // P0 holds the card with 6
    lay_road(s, 1, 27, 32);             // P1 and P2 tie at 5
    lay_road(s, 2, 47, 52);
    s.edge[edge_between(11, 21)] = 3;   // P3 reaches node 11 on P0's road
    s.longest_road_owner = 0;
    give_vp(s, 0, 2);

    act(s, b, action::SETTLE_BASE + 11, name);  // splits P0 into 4 + 2
    expect(s.player_road_length[0] == 4, name, "P0 road should be cut to 4");
    expect(s.longest_road_owner == NO_PLAYER, name, "tied runners-up must not receive the card");
    expect(s.player_vp[0] == 0, name, "former holder must lose 2 VP");
    expect(s.player_vp[1] == 0 && s.player_vp[2] == 0, name, "no tied player may gain VP");

    act(s, b, road(21, 22), name);       // any later check keeps the tie unresolved
    expect(s.longest_road_owner == NO_PLAYER, name, "a later check must not hand out a tied card");

    s.current_player = 1;
    act(s, b, road(32, 33), name);       // P1 alone is longest with 6
    expect(s.longest_road_owner == 1 && s.player_vp[1] == 2, name,
           "unique longest road must claim the card");

    s.current_player = 2;
    act(s, b, road(52, 53), name);       // P2 only equals the holder
    expect(s.player_road_length[2] == 6 && s.longest_road_owner == 1, name,
           "an equal-length road must not take the card");
}

void longest_road_holder_keeps_shared_lead() {
    const char* name = "longest road holder tie";
    GameState s; BoardLayout b;
    empty_main(s, b, 3);
    lay_road(s, 0, 16, 23);             // P0 holds the card with 7
    lay_road(s, 1, 38, 43);             // P1 has 5
    s.edge[edge_between(18, 29)] = 3;
    s.longest_road_owner = 0;
    give_vp(s, 0, 2);

    act(s, b, action::SETTLE_BASE + 18, name);  // P0 left with 2 + 5
    expect(s.player_road_length[0] == 5, name, "P0 road should be cut to 5");
    expect(s.longest_road_owner == 0 && s.player_vp[0] == 2, name,
           "holder still tied for longest must keep the card");

    s.current_player = 1;
    act(s, b, road(43, 44), name);       // P1 now strictly longer
    expect(s.longest_road_owner == 1 && s.player_vp[1] == 2 && s.player_vp[0] == 0, name,
           "a strictly longer road must take the card");
}

void road_length_ignores_build_order() {
    const char* name = "road build order";
    // Same final board, built in both orders: P0 roads 16..21 and 21-22,
    // P1 settlement on node 21.
    GameState before; BoardLayout b;
    empty_main(before, b, 0);
    lay_road(before, 0, 16, 20);
    before.node[21] = node_pack(NODE_SETTLEMENT, 1);
    before.edge[edge_between(21, 22)] = 0;
    act(before, b, road(20, 21), name);  // P0 builds into the settlement
    expect(before.player_road_length[0] == 5, name,
           "edge into an opponent settlement must count, not pass through it");
    expect(!legal(before, b, road(11, 21)), name,
           "P0 must not build through P1's settlement");
    expect(legal(before, b, road(22, 23)), name,
           "P0 may extend its own road behind the settlement");
    GameState owner_view = before;
    owner_view.current_player = 1;
    expect(legal(owner_view, b, road(11, 21)), name,
           "P1 may build from its own settlement");

    GameState after;
    empty_main(after, b, 1);
    lay_road(after, 0, 16, 21);          // P0 reaches node 21 first
    after.edge[edge_between(11, 21)] = 1;
    act(after, b, action::SETTLE_BASE + 21, name);  // P1 settles on P0's road end
    expect(after.player_road_length[0] == 5, name,
           "road length must not depend on build order");
    after.current_player = 0;
    expect(!legal(after, b, road(21, 22)), name,
           "reaching a node first must not let P0 build through a later settlement");
}

void win_only_on_own_turn() {
    const char* name = "own-turn win";
    GameState s; BoardLayout b;
    empty_main(s, b, 0);
    give_vp(s, 0, 9);
    give_vp(s, 1, 10);
    give_vp(s, 2, 11);

    bool done = act(s, b, action::TRADE_ADD_WANT_BASE, name);
    expect(!done && winner_of(s) == NO_PLAYER, name,
           "others reaching 10 must not end P0's turn");

    done = act(s, b, action::END_TURN, name);
    expect(done && winner_of(s) == 1, name,
           "P1 must win when their turn starts, ahead of higher-VP P2");
}

void builder_wins_over_longest_road_beneficiary() {
    const char* name = "simultaneous 10 VP";
    GameState s; BoardLayout b;
    empty_main(s, b, 3);
    lay_road(s, 0, 7, 13);              // P0 holds the card with 6
    lay_road(s, 1, 27, 32);             // P1 is the unique runner-up with 5
    s.edge[edge_between(11, 21)] = 3;
    s.longest_road_owner = 0;
    give_vp(s, 0, 2);
    give_vp(s, 1, 8);
    give_vp(s, 3, 9);

    bool done = act(s, b, action::SETTLE_BASE + 11, name);
    expect(s.longest_road_owner == 1 && s.player_vp[1] == 10, name,
           "the cut must hand the card to P1");
    expect(done && winner_of(s) == 3, name,
           "P3 reached 10 on their own turn and must be the winner");
}

void bank_shortage_single_recipient() {
    const char* name = "bank shortage";
    GameState base; BoardLayout b;
    empty_main(base, b, 0);
    base.dice_roll = 0;  // pre-roll

    // 2 and 12 each appear on exactly one hex.
    uint8_t hex = topology::NUM_HEXES;
    for (uint8_t h = 0; h < topology::NUM_HEXES; ++h)
        if (b.hex_number[h] == 2) hex = h;
    expect(hex != topology::NUM_HEXES && hex != base.robber_hex, name, "test needs the 2 hex");
    const uint8_t r = b.hex_resource[hex];
    const uint8_t city = topology::hex_to_node[hex][0];
    const uint8_t settlement = topology::hex_to_node[hex][2];

    auto roll_two = [&](const GameState& start, uint8_t bank) {
        GameState s = start;
        s.player_resources[3][r] += uint8_t(s.bank[r] - bank);  // keep 19 in play
        s.player_handsize[3] += uint8_t(s.bank[r] - bank);
        s.bank[r] = bank;
        GameState outcomes[MAX_ACTION_OUTCOMES];
        double probabilities[MAX_ACTION_OUTCOMES];
        enumerate_action_outcomes(s, b, action::ROLL_DICE, outcomes, probabilities);
        return outcomes[0];  // dice total 2
    };

    GameState one = base;
    one.node[city] = node_pack(NODE_CITY, 0);  // owed 2
    GameState rolled = roll_two(one, 1);
    expect(rolled.player_resources[0][r] == 5 && rolled.bank[r] == 0, name,
           "a single recipient must take what the bank has left");

    GameState two = one;
    two.node[settlement] = node_pack(NODE_SETTLEMENT, 1);  // owed 1 more
    rolled = roll_two(two, 2);
    expect(rolled.player_resources[0][r] == 4 && rolled.player_resources[1][r] == 4
               && rolled.bank[r] == 2, name,
           "a shortage shared by several players must pay nobody");

    rolled = roll_two(two, 3);
    expect(rolled.player_resources[0][r] == 6 && rolled.player_resources[1][r] == 5
               && rolled.bank[r] == 0, name,
           "a bank that covers everyone must pay in full");
}

}  // namespace

int main() {
    longest_road_tie_is_set_aside();
    longest_road_holder_keeps_shared_lead();
    road_length_ignores_build_order();
    win_only_on_own_turn();
    builder_wins_over_longest_road_beneficiary();
    bank_shortage_single_recipient();

    if (failures) {
        std::fprintf(stderr, "%d rule check(s) failed\n", failures);
        return 1;
    }
    std::printf("rule scenarios passed\n");
    return 0;
}

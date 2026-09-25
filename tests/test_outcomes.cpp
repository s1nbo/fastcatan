#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "mask.hpp"
#include "rules.hpp"
#include "state.hpp"

using namespace catan;

namespace {

uint32_t first_legal(const GameState& state) {
    for (uint32_t action = 0; action < NUM_ACTIONS; ++action) {
        if (state.action_mask[action >> 6]
            & (uint64_t(1) << (action & 63))) {
            return action;
        }
    }
    return NUM_ACTIONS;
}

int fail(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    return 1;
}

}  // namespace

int main() {
    GameState state{};
    BoardLayout layout{};
    reset_one(state, layout, 0);

    GameState outcomes[MAX_ACTION_OUTCOMES];
    double probabilities[MAX_ACTION_OUTCOMES];
    GameState original = state;

    uint32_t count = enumerate_action_outcomes(
        state, layout, action::ROLL_DICE, outcomes, probabilities);
    if (count != 1 || probabilities[0] != 1.0
        || std::memcmp(&outcomes[0], &state, sizeof(GameState)) != 0) {
        return fail("illegal action did not produce one identity outcome");
    }

    uint32_t action_id = first_legal(state);
    count = enumerate_action_outcomes(
        state, layout, action_id, outcomes, probabilities);
    GameState direct = state;
    step_one(direct, layout, action_id);
    if (count != 1 || probabilities[0] != 1.0
        || std::memcmp(&outcomes[0], &direct, sizeof(GameState)) != 0
        || std::memcmp(&state, &original, sizeof(GameState)) != 0) {
        return fail("deterministic outcome diverged from sampled step");
    }

    state = direct;
    while (state.phase != Phase::MAIN) {
        action_id = first_legal(state);
        if (action_id == NUM_ACTIONS)
            return fail("initial placement had no legal action");
        step_one(state, layout, action_id);
    }

    original = state;
    count = enumerate_action_outcomes(
        state, layout, action::ROLL_DICE, outcomes, probabilities);
    if (count != MAX_ACTION_OUTCOMES)
        return fail("dice action did not produce eleven outcomes");

    double total = 0.0;
    constexpr int ways[11] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    for (uint32_t index = 0; index < count; ++index) {
        total += probabilities[index];
        if (outcomes[index].dice_roll != index + 2
            || probabilities[index] != double(ways[index]) / 36.0) {
            return fail("dice outcome or probability is incorrect");
        }
        uint64_t recomputed[MASK_WORDS];
        compute_mask(outcomes[index], layout, recomputed);
        if (std::memcmp(recomputed, outcomes[index].action_mask,
                        sizeof(recomputed)) != 0) {
            return fail("enumerated outcome has a stale legal-action mask");
        }
    }
    GameState sampled = state;
    step_one(sampled, layout, action::ROLL_DICE);
    bool sampled_outcome_found = false;
    for (uint32_t index = 0; index < count; ++index) {
        if (std::memcmp(&sampled, &outcomes[index], sizeof(GameState)) == 0) {
            sampled_outcome_found = true;
            break;
        }
    }
    if (!sampled_outcome_found)
        return fail("sampled dice transition is absent from exact outcomes");
    if (std::abs(total - 1.0) > 1e-12
        || std::memcmp(&state, &original, sizeof(GameState)) != 0) {
        return fail("enumeration changed its source or probabilities do not sum to one");
    }

    return 0;
}

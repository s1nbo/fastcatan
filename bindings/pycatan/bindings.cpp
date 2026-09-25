// Python bindings for the FastCatan simulator.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "batched_env.hpp"
#include "mask.hpp"
#include "rules.hpp"
#include "topology.hpp"

namespace nb = nanobind;
using namespace catan;

namespace {

inline constexpr uint32_t STATE_VIEW_VERSION = 1;

void check_index(uint32_t value, uint32_t limit, const char* name) {
    if (value >= limit)
        throw std::runtime_error(std::string(name) + " out of range");
}

void validate_snapshot_bytes(const char* data) {
    auto* bytes = reinterpret_cast<const unsigned char*>(data);
    if (bytes[offsetof(GameState, dev_card_played)] > 1)
        throw std::runtime_error("invalid snapshot state");
}

void validate_snapshot(const GameState& s, const BoardLayout& b) {
    if (uint8_t(s.phase) > uint8_t(Phase::ENDED)
        || uint8_t(s.flag) > uint8_t(Flag::TRADE_PENDING)
        || s.current_player >= NUM_PLAYERS
        || s.start_player >= NUM_PLAYERS
        || s.discarding_player >= NUM_PLAYERS
        || s.robber_hex >= topology::NUM_HEXES
        || (s.longest_road_owner != NO_PLAYER
            && s.longest_road_owner >= NUM_PLAYERS)
        || (s.largest_army_owner != NO_PLAYER
            && s.largest_army_owner >= NUM_PLAYERS)
        || (s.trade_proposer != NO_PLAYER
            && s.trade_proposer >= NUM_PLAYERS)
        || s.free_roads_remaining > 2
        || (s.dice_roll != 0 && (s.dice_roll < 2 || s.dice_roll > 12))) {
        throw std::runtime_error("invalid snapshot state");
    }

    for (uint32_t i = 0; i < topology::NUM_NODES; ++i) {
        uint8_t node = s.node[i];
        uint8_t level = node_level(node);
        if ((node & 0xF0) != 0 || level > NODE_CITY
            || (level == NODE_EMPTY && node != NODE_EMPTY)
            || (level != NODE_EMPTY && node_owner(node) >= NUM_PLAYERS)) {
            throw std::runtime_error("invalid snapshot node");
        }
    }
    for (uint32_t i = 0; i < topology::NUM_EDGES; ++i) {
        if (s.edge[i] != NO_PLAYER && s.edge[i] >= NUM_PLAYERS)
            throw std::runtime_error("invalid snapshot edge");
    }
    for (uint32_t h = 0; h < topology::NUM_HEXES; ++h) {
        uint8_t resource = b.hex_resource[h];
        uint8_t number = b.hex_number[h];
        if (resource > NUM_RESOURCES
            || (resource == NUM_RESOURCES && number != 0)
            || (resource < NUM_RESOURCES
                && (number < 2 || number > 12 || number == 7))) {
            throw std::runtime_error("invalid snapshot hex");
        }
    }
    for (uint32_t p = 0; p < topology::NUM_PORTS; ++p) {
        if (b.port_type[p] > NUM_RESOURCES)
            throw std::runtime_error("invalid snapshot port");
    }

    for (uint32_t player = 0; player < NUM_PLAYERS; ++player) {
        uint32_t handsize = 0;
        for (uint32_t resource = 0; resource < NUM_RESOURCES; ++resource)
            handsize += s.player_resources[player][resource];
        if (handsize != s.player_handsize[player]
            || s.player_settlement_count[player] > 5
            || s.player_city_count[player] > 4
            || s.player_road_count[player] > 15
            || s.player_discard_remaining[player] > s.player_handsize[player]) {
            throw std::runtime_error("invalid snapshot player state");
        }
    }
    for (uint32_t resource = 0; resource < NUM_RESOURCES; ++resource) {
        uint32_t total = s.bank[resource];
        for (uint32_t player = 0; player < NUM_PLAYERS; ++player)
            total += s.player_resources[player][resource];
        if (total != 19)
            throw std::runtime_error("invalid snapshot resource total");
    }
}

uint8_t winner_of(const GameState& state) {
    for (uint8_t player = 0; player < NUM_PLAYERS; ++player) {
        if (state.player_vp[player] >= WIN_VP) return player;
    }
    return NO_PLAYER;
}

nb::bytes snapshot_bytes(const GameState& state, const BoardLayout& layout) {
    char data[SNAPSHOT_BYTES];
    std::memcpy(data, &state, sizeof(GameState));
    std::memcpy(data + sizeof(GameState), &layout, sizeof(BoardLayout));
    return nb::bytes(data, sizeof(data));
}

std::vector<uint32_t> legal_actions_of(const GameState& state) {
    std::vector<uint32_t> actions;
    for (uint32_t action = 0; action < NUM_ACTIONS; ++action) {
        if (state.action_mask[action >> 6]
            & (uint64_t(1) << (action & 63))) {
            actions.push_back(action);
        }
    }
    return actions;
}

template <typename Value>
nb::list integer_list(uint32_t size, Value value) {
    nb::list result;
    for (uint32_t index = 0; index < size; ++index)
        result.append(nb::int_(value(index)));
    return result;
}

nb::dict state_view(const GameState& state, const BoardLayout& layout,
                    uint8_t pov) {
    nb::dict view;
    view["schema_version"] = nb::int_(STATE_VIEW_VERSION);
    view["pov"] = nb::int_(pov);
    view["phase"] = nb::int_(uint8_t(state.phase));
    view["flag"] = nb::int_(uint8_t(state.flag));
    view["current_player"] = nb::int_(state.current_player);
    view["discarding_player"] = nb::int_(state.discarding_player);
    view["actor_to_act"] = nb::int_(actor_to_act(state));
    view["dice_roll"] = nb::int_(state.dice_roll);
    view["turn_count"] = nb::int_(state.turn_count);
    view["done"] = nb::bool_(state.phase == Phase::ENDED);
    view["winner"] = nb::int_(winner_of(state));
    view["robber_hex"] = nb::int_(state.robber_hex);
    view["start_player"] = nb::int_(state.start_player);
    view["free_roads_remaining"] = nb::int_(state.free_roads_remaining);
    view["dev_card_played"] = nb::bool_(state.dev_card_played);
    view["longest_road_owner"] = nb::int_(state.longest_road_owner);
    view["largest_army_owner"] = nb::int_(state.largest_army_owner);

    nb::list players;
    for (uint8_t seat = 0; seat < NUM_PLAYERS; ++seat) {
        bool self = seat == pov;
        nb::dict player;
        player["seat"] = nb::int_(seat);
        player["victory_points"] = nb::int_(
            self ? state.player_vp[seat]
                 : state.player_vp_without_dev[seat]);
        player["public_victory_points"] =
            nb::int_(state.player_vp_without_dev[seat]);
        player["hand_size"] = nb::int_(state.player_handsize[seat]);
        player["development_card_count"] =
            nb::int_(state.player_total_dev[seat]);
        player["knights_played"] =
            nb::int_(state.player_knights_played[seat]);
        player["road_length"] = nb::int_(state.player_road_length[seat]);
        player["settlements_remaining"] =
            nb::int_(state.player_settlement_count[seat]);
        player["cities_remaining"] =
            nb::int_(state.player_city_count[seat]);
        player["roads_remaining"] = nb::int_(state.player_road_count[seat]);
        player["ports"] = nb::int_(state.player_ports[seat]);
        player["discard_remaining"] =
            nb::int_(state.player_discard_remaining[seat]);
        if (self) {
            player["resources"] = integer_list(
                NUM_RESOURCES,
                [&](uint32_t resource) {
                    return state.player_resources[seat][resource];
                });
            player["development_cards"] = integer_list(
                5, [&](uint32_t card) { return state.player_dev[seat][card]; });
            player["development_cards_bought_this_turn"] = integer_list(
                5, [&](uint32_t card) {
                    return state.player_dev_bought_this_turn[seat][card];
                });
        } else {
            player["resources"] = nb::none();
            player["development_cards"] = nb::none();
            player["development_cards_bought_this_turn"] = nb::none();
        }
        players.append(player);
    }
    view["players"] = players;

    view["bank"] = integer_list(
        NUM_RESOURCES,
        [&](uint32_t resource) { return state.bank[resource]; });
    uint32_t development_deck_size = 0;
    for (uint8_t card = 0; card < 5; ++card)
        development_deck_size += state.dev_deck[card];
    view["development_deck_size"] = nb::int_(development_deck_size);

    view["node_levels"] = integer_list(
        topology::NUM_NODES,
        [&](uint32_t node) { return node_level(state.node[node]); });
    view["node_owners"] = integer_list(
        topology::NUM_NODES, [&](uint32_t node) {
            return node_level(state.node[node]) == NODE_EMPTY
                ? NO_PLAYER
                : node_owner(state.node[node]);
        });
    view["edge_owners"] = integer_list(
        topology::NUM_EDGES,
        [&](uint32_t edge) { return state.edge[edge]; });
    view["hex_resources"] = integer_list(
        topology::NUM_HEXES,
        [&](uint32_t hex) { return layout.hex_resource[hex]; });
    view["hex_numbers"] = integer_list(
        topology::NUM_HEXES,
        [&](uint32_t hex) { return layout.hex_number[hex]; });
    view["port_types"] = integer_list(
        topology::NUM_PORTS,
        [&](uint32_t port) { return layout.port_type[port]; });

    bool trade_open = state.trade_proposer != NO_PLAYER;
    bool own_draft = !trade_open && pov == state.current_player;
    nb::dict trade;
    trade["proposer"] = nb::int_(
        trade_open ? state.trade_proposer : NO_PLAYER);
    if (trade_open || own_draft) {
        trade["give"] = integer_list(
            NUM_RESOURCES,
            [&](uint32_t resource) { return state.trade_give[resource]; });
        trade["want"] = integer_list(
            NUM_RESOURCES,
            [&](uint32_t resource) { return state.trade_want[resource]; });
    } else {
        trade["give"] = nb::none();
        trade["want"] = nb::none();
    }
    if (trade_open) {
        trade["responses"] = integer_list(
            NUM_PLAYERS, [&](uint32_t seat) {
                return uint8_t((state.trade_response >> (seat * 2)) & 0x03);
            });
    } else {
        trade["responses"] = nb::none();
    }
    view["trade"] = trade;

    if (pov == actor_to_act(state))
        view["legal_actions"] = nb::cast(legal_actions_of(state));
    else
        view["legal_actions"] = nb::none();
    return view;
}

struct PyBatchedEnv {
    BatchedEnv inner{};
    bool initialized = false;

    PyBatchedEnv(uint32_t n_envs, uint64_t seed) {
        if (n_envs == 0
            || n_envs > uint32_t(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("num_envs out of range");
        }
        if (!batched_env_init(inner, n_envs, seed))
            throw std::bad_alloc();
    }
    ~PyBatchedEnv() { batched_env_destroy(inner); }
    PyBatchedEnv(const PyBatchedEnv&) = delete;
    PyBatchedEnv& operator=(const PyBatchedEnv&) = delete;

    void require_initialized() const {
        if (!initialized)
            throw std::runtime_error("environment is not reset");
    }
    void check_env(uint32_t env_idx) const {
        require_initialized();
        check_index(env_idx, inner.n, "env_idx");
    }
};

struct PyEnv {
    GameState s{};
    BoardLayout b{};
    bool initialized = false;

    void require_initialized() const {
        if (!initialized)
            throw std::runtime_error("environment is not reset");
    }
    void check_seat(uint32_t seat) const {
        require_initialized();
        check_index(seat, NUM_PLAYERS, "seat");
    }
    void check_resource(uint32_t resource) const {
        require_initialized();
        check_index(resource, NUM_RESOURCES, "resource");
    }

    void reset(uint64_t seed) noexcept {
        reset_one(s, b, seed);
        initialized = true;
    }
    bool step(uint32_t action) {
        require_initialized();
        return step_one(s, b, action);
    }
    uint8_t phase() const { require_initialized(); return uint8_t(s.phase); }
    uint8_t flag() const { require_initialized(); return uint8_t(s.flag); }
    uint8_t current_player() const { require_initialized(); return s.current_player; }
    uint8_t discarding_player() const { require_initialized(); return s.discarding_player; }
    uint8_t actor() const { require_initialized(); return actor_to_act(s); }
    uint8_t dice_roll() const { require_initialized(); return s.dice_roll; }
    uint16_t turn_count() const { require_initialized(); return s.turn_count; }
    bool done() const { require_initialized(); return s.phase == Phase::ENDED; }
    uint8_t winner() const {
        require_initialized();
        return winner_of(s);
    }

    uint8_t player_vp(uint32_t seat) const { check_seat(seat); return s.player_vp[seat]; }
    uint8_t player_vp_public(uint32_t seat) const { check_seat(seat); return s.player_vp_without_dev[seat]; }
    uint8_t player_handsize(uint32_t seat) const { check_seat(seat); return s.player_handsize[seat]; }
    uint8_t player_settlement_count(uint32_t seat) const { check_seat(seat); return s.player_settlement_count[seat]; }
    uint8_t player_city_count(uint32_t seat) const { check_seat(seat); return s.player_city_count[seat]; }
    uint8_t player_road_count(uint32_t seat) const { check_seat(seat); return s.player_road_count[seat]; }
    uint8_t player_knights_played(uint32_t seat) const { check_seat(seat); return s.player_knights_played[seat]; }
    uint8_t player_road_length(uint32_t seat) const { check_seat(seat); return s.player_road_length[seat]; }
    uint8_t player_ports(uint32_t seat) const { check_seat(seat); return s.player_ports[seat]; }
    uint8_t player_total_dev(uint32_t seat) const { check_seat(seat); return s.player_total_dev[seat]; }
    uint8_t player_discard_remaining(uint32_t seat) const { check_seat(seat); return s.player_discard_remaining[seat]; }
    uint8_t player_resource(uint32_t seat, uint32_t resource) const {
        check_seat(seat);
        check_resource(resource);
        return s.player_resources[seat][resource];
    }
    uint8_t player_dev(uint32_t seat, uint32_t card) const {
        check_seat(seat);
        check_index(card, 5, "card");
        return s.player_dev[seat][card];
    }
    uint8_t player_dev_bought(uint32_t seat, uint32_t card) const {
        check_seat(seat);
        check_index(card, 5, "card");
        return s.player_dev_bought_this_turn[seat][card];
    }
    uint8_t bank(uint32_t resource) const { check_resource(resource); return s.bank[resource]; }
    uint8_t dev_deck(uint32_t card) const {
        require_initialized();
        check_index(card, 5, "card");
        return s.dev_deck[card];
    }
    uint8_t node(uint32_t node_id) const {
        require_initialized();
        check_index(node_id, topology::NUM_NODES, "node");
        return s.node[node_id];
    }
    uint8_t edge(uint32_t edge_id) const {
        require_initialized();
        check_index(edge_id, topology::NUM_EDGES, "edge");
        return s.edge[edge_id];
    }
    uint8_t hex_resource(uint32_t hex) const {
        require_initialized();
        check_index(hex, topology::NUM_HEXES, "hex");
        return b.hex_resource[hex];
    }
    uint8_t hex_number(uint32_t hex) const {
        require_initialized();
        check_index(hex, topology::NUM_HEXES, "hex");
        return b.hex_number[hex];
    }
    uint8_t port_type(uint32_t port) const {
        require_initialized();
        check_index(port, topology::NUM_PORTS, "port");
        return b.port_type[port];
    }
    uint8_t trade_give(uint32_t resource) const { check_resource(resource); return s.trade_give[resource]; }
    uint8_t trade_want(uint32_t resource) const { check_resource(resource); return s.trade_want[resource]; }
    uint8_t trade_response(uint32_t seat) const {
        check_seat(seat);
        return uint8_t((s.trade_response >> (seat * 2)) & 0x03);
    }

    std::vector<uint32_t> legal_actions() const {
        require_initialized();
        return legal_actions_of(s);
    }

    nb::bytes snapshot() const {
        require_initialized();
        return snapshot_bytes(s, b);
    }
    void load_snapshot(nb::bytes data) {
        if (data.size() != SNAPSHOT_BYTES)
            throw std::runtime_error("snapshot size mismatch");
        validate_snapshot_bytes(data.c_str());
        GameState next_state{};
        BoardLayout next_layout{};
        std::memcpy(&next_state, data.c_str(), sizeof(GameState));
        std::memcpy(&next_layout, data.c_str() + sizeof(GameState),
                    sizeof(BoardLayout));
        validate_snapshot(next_state, next_layout);
        compute_mask(next_state, next_layout, next_state.action_mask);
        s = next_state;
        b = next_layout;
        initialized = true;
    }
    void reseed(uint64_t seed) {
        require_initialized();
        xoshiro_seed(s.rng, seed);
    }
    void action_mask(
        nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig,
                    nb::device::cpu> out) const {
        require_initialized();
        if (out.shape(0) != MASK_WORDS)
            throw std::runtime_error("action_mask buffer length mismatch");
        std::memcpy(out.data(), s.action_mask, sizeof(s.action_mask));
    }

    nb::list enumerate_outcomes(uint32_t action) const {
        require_initialized();
        GameState states[MAX_ACTION_OUTCOMES];
        double probabilities[MAX_ACTION_OUTCOMES];
        uint32_t count = enumerate_action_outcomes(
            s, b, action, states, probabilities);
        nb::list outcomes;
        for (uint32_t index = 0; index < count; ++index) {
            outcomes.append(nb::make_tuple(
                probabilities[index], snapshot_bytes(states[index], b)));
        }
        return outcomes;
    }

    nb::dict view(uint32_t pov) const {
        require_initialized();
        check_index(pov, NUM_PLAYERS, "pov");
        return state_view(s, b, uint8_t(pov));
    }
};

}  // namespace

NB_MODULE(_fastcatan, m) {
    m.doc() = "FastCatan game simulator (C++ core via nanobind).";

    m.attr("MASK_WORDS") = MASK_WORDS;
    m.attr("NUM_ACTIONS") = NUM_ACTIONS;
    m.attr("NUM_PLAYERS") = uint32_t(NUM_PLAYERS);
    m.attr("NUM_RESOURCES") = uint32_t(NUM_RESOURCES);
    m.attr("NUM_NODES") = uint32_t(topology::NUM_NODES);
    m.attr("NUM_EDGES") = uint32_t(topology::NUM_EDGES);
    m.attr("NUM_HEXES") = uint32_t(topology::NUM_HEXES);
    m.attr("NUM_PORTS") = uint32_t(topology::NUM_PORTS);
    m.attr("NO_PLAYER") = uint32_t(NO_PLAYER);
    m.attr("NO_ACTION") = NO_ACTION;
    m.attr("SNAPSHOT_BYTES") = uint32_t(SNAPSHOT_BYTES);
    m.attr("MAX_ACTION_OUTCOMES") = MAX_ACTION_OUTCOMES;
    m.attr("STATE_VIEW_VERSION") = STATE_VIEW_VERSION;

    nb::module_ act = m.def_submodule("action", "Flat action ID layout.");
    act.attr("SETTLE_BASE") = action::SETTLE_BASE;
    act.attr("CITY_BASE") = action::CITY_BASE;
    act.attr("ROAD_BASE") = action::ROAD_BASE;
    act.attr("ROLL_DICE") = action::ROLL_DICE;
    act.attr("END_TURN") = action::END_TURN;
    act.attr("DISCARD_BASE") = action::DISCARD_BASE;
    act.attr("MOVE_ROBBER_BASE") = action::MOVE_ROBBER_BASE;
    act.attr("STEAL_BASE") = action::STEAL_BASE;
    act.attr("TRADE_BASE") = action::TRADE_BASE;
    act.attr("BUY_DEV") = action::BUY_DEV;
    act.attr("PLAY_KNIGHT") = action::PLAY_KNIGHT;
    act.attr("PLAY_ROAD_BUILDING") = action::PLAY_ROAD_BUILDING;
    act.attr("PLAY_YEAR_OF_PLENTY") = action::PLAY_YEAR_OF_PLENTY;
    act.attr("PLAY_MONOPOLY") = action::PLAY_MONOPOLY;
    act.attr("TRADE_ADD_GIVE_BASE") = action::TRADE_ADD_GIVE_BASE;
    act.attr("TRADE_ADD_WANT_BASE") = action::TRADE_ADD_WANT_BASE;
    act.attr("TRADE_OPEN") = action::TRADE_OPEN;
    act.attr("TRADE_ACCEPT") = action::TRADE_ACCEPT;
    act.attr("TRADE_DECLINE") = action::TRADE_DECLINE;
    act.attr("TRADE_CONFIRM_BASE") = action::TRADE_CONFIRM_BASE;
    act.attr("TRADE_CANCEL") = action::TRADE_CANCEL;

    nb::class_<PyEnv>(m, "Env", "One Catan game.")
        .def(nb::init<>(), "Construct an uninitialized game; call reset next.")
        .def("reset", &PyEnv::reset, nb::arg("seed"))
        .def("step", &PyEnv::step, nb::arg("action"),
             "Apply an action and return whether the game is terminal. Illegal actions are no-ops.")
        .def_prop_ro("phase", &PyEnv::phase)
        .def_prop_ro("flag", &PyEnv::flag)
        .def_prop_ro("current_player", &PyEnv::current_player)
        .def_prop_ro("discarding_player", &PyEnv::discarding_player)
        .def_prop_ro("actor_to_act", &PyEnv::actor)
        .def_prop_ro("dice_roll", &PyEnv::dice_roll)
        .def_prop_ro("turn_count", &PyEnv::turn_count)
        .def_prop_ro("done", &PyEnv::done)
        .def_prop_ro("winner", &PyEnv::winner,
                     "Winning seat, or NO_PLAYER if the game has no winner.")
        .def_prop_ro("robber_hex", [](const PyEnv& e) { e.require_initialized(); return e.s.robber_hex; })
        .def_prop_ro("start_player", [](const PyEnv& e) { e.require_initialized(); return e.s.start_player; })
        .def_prop_ro("free_roads_remaining", [](const PyEnv& e) { e.require_initialized(); return e.s.free_roads_remaining; })
        .def_prop_ro("dev_card_played", [](const PyEnv& e) { e.require_initialized(); return e.s.dev_card_played; })
        .def_prop_ro("longest_road_owner", [](const PyEnv& e) { e.require_initialized(); return e.s.longest_road_owner; })
        .def_prop_ro("largest_army_owner", [](const PyEnv& e) { e.require_initialized(); return e.s.largest_army_owner; })
        .def_prop_ro("trade_proposer", [](const PyEnv& e) { e.require_initialized(); return e.s.trade_proposer; })
        .def_prop_ro("trade_compose_count", [](const PyEnv& e) { e.require_initialized(); return e.s.trade_compose_count; })
        .def("player_vp", &PyEnv::player_vp, nb::arg("seat"))
        .def("player_vp_public", &PyEnv::player_vp_public, nb::arg("seat"))
        .def("player_handsize", &PyEnv::player_handsize, nb::arg("seat"))
        .def("player_settlement_count", &PyEnv::player_settlement_count, nb::arg("seat"))
        .def("player_city_count", &PyEnv::player_city_count, nb::arg("seat"))
        .def("player_road_count", &PyEnv::player_road_count, nb::arg("seat"))
        .def("player_knights_played", &PyEnv::player_knights_played, nb::arg("seat"))
        .def("player_road_length", &PyEnv::player_road_length, nb::arg("seat"))
        .def("player_ports", &PyEnv::player_ports, nb::arg("seat"))
        .def("player_total_dev", &PyEnv::player_total_dev, nb::arg("seat"))
        .def("player_discard_remaining", &PyEnv::player_discard_remaining, nb::arg("seat"))
        .def("player_resource", &PyEnv::player_resource, nb::arg("seat"), nb::arg("resource"))
        .def("player_dev", &PyEnv::player_dev, nb::arg("seat"), nb::arg("card"))
        .def("player_dev_bought", &PyEnv::player_dev_bought, nb::arg("seat"), nb::arg("card"))
        .def("bank", &PyEnv::bank, nb::arg("resource"))
        .def("dev_deck", &PyEnv::dev_deck, nb::arg("card"))
        .def("node", &PyEnv::node, nb::arg("node"))
        .def("edge", &PyEnv::edge, nb::arg("edge"))
        .def("hex_resource", &PyEnv::hex_resource, nb::arg("hex"))
        .def("hex_number", &PyEnv::hex_number, nb::arg("hex"))
        .def("port_type", &PyEnv::port_type, nb::arg("port"))
        .def("trade_give", &PyEnv::trade_give, nb::arg("resource"))
        .def("trade_want", &PyEnv::trade_want, nb::arg("resource"))
        .def("trade_response", &PyEnv::trade_response, nb::arg("seat"))
        .def("legal_actions", &PyEnv::legal_actions,
             "Return the currently legal action IDs.")
        .def("action_mask", &PyEnv::action_mask, nb::arg("out"),
             "Copy the legal-action bitset into a uint64[MASK_WORDS] array.")
        .def("enumerate_outcomes", &PyEnv::enumerate_outcomes,
             nb::arg("action"),
             "Return exact physical outcomes as (probability, snapshot) pairs. "
             "Deterministic and illegal actions produce one outcome.")
        .def("state_view", &PyEnv::view, nb::arg("pov"),
             "Return a semantic state dictionary that exposes private cards "
             "only for the requested player.")
        .def("snapshot", &PyEnv::snapshot)
        .def("load_snapshot", &PyEnv::load_snapshot, nb::arg("data"),
             "Restore and validate a snapshot produced by this package version.")
        .def("reseed", &PyEnv::reseed, nb::arg("seed"));

    using ArrU32 = nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrU8 = nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrConstU8_2D = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    using ArrU8_2D = nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    using ArrU64 = nb::ndarray<const uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrU64_2D = nb::ndarray<uint64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

    nb::class_<PyBatchedEnv>(m, "BatchedEnv",
        "A contiguous collection of independent Catan games.")
        .def(nb::init<uint32_t, uint64_t>(), nb::arg("num_envs"), nb::arg("seed") = 42)
        .def_prop_ro("num_envs", [](const PyBatchedEnv& e) { return e.inner.n; })
        .def("reset", [](PyBatchedEnv& e) {
            {
                nb::gil_scoped_release release;
                batched_env_reset(e.inner);
            }
            e.initialized = true;
        })
        .def("step", [](PyBatchedEnv& e, ArrU32 actions, ArrU8 dones) {
            e.require_initialized();
            if (actions.shape(0) != e.inner.n || dones.shape(0) != e.inner.n)
                throw std::runtime_error("step buffer length mismatch");
            nb::gil_scoped_release release;
            batched_env_step(e.inner, actions.data(), dones.data());
        }, nb::arg("actions"), nb::arg("dones_out"),
        "Step every game and preserve terminal states.")
        .def("step_autoreset", [](PyBatchedEnv& e, ArrU32 actions, ArrU8 dones) {
            e.require_initialized();
            if (actions.shape(0) != e.inner.n || dones.shape(0) != e.inner.n)
                throw std::runtime_error("step buffer length mismatch");
            nb::gil_scoped_release release;
            batched_env_step_autoreset(e.inner, actions.data(), dones.data());
        }, nb::arg("actions"), nb::arg("dones_out"),
        "Step every game and immediately reset terminal slots. Read "
        "last_winner after a reported terminal transition.")
        .def("write_masks", [](const PyBatchedEnv& e, ArrU64_2D out) {
            e.require_initialized();
            if (out.shape(0) != e.inner.n || out.shape(1) != MASK_WORDS)
                throw std::runtime_error("mask buffer shape mismatch");
            nb::gil_scoped_release release;
            batched_env_write_masks(e.inner, out.data());
        }, nb::arg("out"))
        .def("phase", [](const PyBatchedEnv& e, uint32_t env_idx) {
            e.check_env(env_idx);
            return uint8_t(e.inner.states[env_idx].phase);
        }, nb::arg("env_idx"))
        .def("current_player", [](const PyBatchedEnv& e, uint32_t env_idx) {
            e.check_env(env_idx);
            return e.inner.states[env_idx].current_player;
        }, nb::arg("env_idx"))
        .def("actor_to_act", [](const PyBatchedEnv& e, uint32_t env_idx) {
            e.check_env(env_idx);
            return ::catan::actor_to_act(e.inner.states[env_idx]);
        }, nb::arg("env_idx"))
        .def("player_vp", [](const PyBatchedEnv& e, uint32_t env_idx, uint32_t player) {
            e.check_env(env_idx);
            check_index(player, NUM_PLAYERS, "player");
            return e.inner.states[env_idx].player_vp[player];
        }, nb::arg("env_idx"), nb::arg("player"))
        .def("player_handsize", [](const PyBatchedEnv& e, uint32_t env_idx, uint32_t player) {
            e.check_env(env_idx);
            check_index(player, NUM_PLAYERS, "player");
            return e.inner.states[env_idx].player_handsize[player];
        }, nb::arg("env_idx"), nb::arg("player"))
        .def("last_winner", [](const PyBatchedEnv& e, uint32_t env_idx) {
            e.check_env(env_idx);
            return e.inner.last_winner[env_idx];
        }, nb::arg("env_idx"))
        .def("state_view", [](const PyBatchedEnv& e, uint32_t env_idx,
                              uint32_t pov) {
            e.check_env(env_idx);
            check_index(pov, NUM_PLAYERS, "pov");
            return state_view(e.inner.states[env_idx], e.inner.layouts[env_idx],
                              uint8_t(pov));
        }, nb::arg("env_idx"), nb::arg("pov"))
        .def("snapshot", [](const PyBatchedEnv& e, uint32_t env_idx) {
            e.check_env(env_idx);
            return snapshot_bytes(e.inner.states[env_idx],
                                  e.inner.layouts[env_idx]);
        }, nb::arg("env_idx"))
        .def("save_snapshots", [](const PyBatchedEnv& e, ArrU8_2D out) {
            e.require_initialized();
            if (out.shape(0) != e.inner.n
                || out.shape(1) != SNAPSHOT_BYTES) {
                throw std::runtime_error("snapshot buffer shape mismatch");
            }
            nb::gil_scoped_release release;
            batched_env_save(e.inner, out.data());
        }, nb::arg("out"))
        .def("load_snapshots", [](PyBatchedEnv& e, ArrConstU8_2D data) {
            if (data.shape(0) != e.inner.n
                || data.shape(1) != SNAPSHOT_BYTES) {
                throw std::runtime_error("snapshot buffer shape mismatch");
            }

            for (uint32_t index = 0; index < e.inner.n; ++index) {
                const uint8_t* row = data.data()
                    + std::size_t(index) * SNAPSHOT_BYTES;
                validate_snapshot_bytes(
                    reinterpret_cast<const char*>(row));
                GameState state{};
                BoardLayout layout{};
                std::memcpy(&state, row, sizeof(GameState));
                std::memcpy(&layout, row + sizeof(GameState),
                            sizeof(BoardLayout));
                validate_snapshot(state, layout);
            }

            // Keep the GIL from validation through the copy. Otherwise another
            // Python thread could mutate a writable NumPy input after it was
            // validated but before the native states were restored.
            batched_env_load(e.inner, data.data());
            e.initialized = true;
        }, nb::arg("data"))
        .def("reseed", [](PyBatchedEnv& e, ArrU64 seeds) {
            e.require_initialized();
            if (seeds.shape(0) != e.inner.n)
                throw std::runtime_error("seeds length mismatch");
            nb::gil_scoped_release release;
            batched_env_reseed(e.inner, seeds.data());
        }, nb::arg("seeds"));
}

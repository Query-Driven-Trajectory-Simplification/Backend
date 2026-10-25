#include "backend/engine/pipeline/config.hpp"

#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>

namespace pipeline {

namespace {

using json = nlohmann::json;

// Parses an enum from its enumerator name, e.g. "RDP"
template <typename E> E parse_enum(const json &value) {
    const auto name = value.get<std::string>();
    const auto parsed = magic_enum::enum_cast<E>(name);
    if (!parsed)
        throw std::invalid_argument("config: unknown value '" + name + "'");
    return *parsed;
}

void parse_split_config(Config &config, const json &split) {
    if (split.contains("type"))
        config.split.type = parse_enum<split::SplitType>(split["type"]);
    config.split.query_percentage = split.value("query_percentage", config.split.query_percentage);
    config.split.seed = split.value("seed", config.split.seed);
}

void parse_knn_config(Config &config, const json &knn) {
    config.knn.k = knn.value("k", config.knn.k);
    config.knn.query_count = knn.value("query_count", config.knn.query_count);
    config.knn.seed = knn.value("seed", config.knn.seed);
}

void parse_range_config(Config &config, const json &range) {
    config.range.query_count = range.value("query_count", config.range.query_count);
    config.range.box_size_m = range.value("box_size_m", config.range.box_size_m);
    config.range.window_seconds = range.value("window_seconds", config.range.window_seconds);
    config.range.min_results = range.value("min_results", config.range.min_results);
    config.range.full_accept_at = range.value("full_accept_at", config.range.full_accept_at);
    config.range.attempts_per_query =
        range.value("attempts_per_query", config.range.attempts_per_query);
    config.range.seed = range.value("seed", config.range.seed);
}

void parse_where_at_config(Config &config, const json &where_at) {
    config.where_at.probes_per_trajectory =
        where_at.value("probes_per_trajectory", config.where_at.probes_per_trajectory);
    config.where_at.seed = where_at.value("seed", config.where_at.seed);
    config.where_at.max_gap_seconds =
        where_at.value("max_gap_seconds", config.where_at.max_gap_seconds);
}

void parse_where_at_scoring_config(Config &config, const json &scoring) {
    config.where_at_scoring.hit_radius_m =
        scoring.value("hit_radius_m", config.where_at_scoring.hit_radius_m);
    config.where_at_scoring.min_tolerance_m =
        scoring.value("min_tolerance_m", config.where_at_scoring.min_tolerance_m);
    config.where_at_scoring.max_tolerance_m =
        scoring.value("max_tolerance_m", config.where_at_scoring.max_tolerance_m);
}

Config parse_config(const json &data) {
    Config config;

    if (!data.contains("data_path"))
        throw std::invalid_argument("config: data_path is required");
    config.data_path = data["data_path"].get<std::string>();

    // Every other key is optional and falls back to the struct's default
    config.budget_shares = data.value("budget_shares", config.budget_shares);
    if (data.contains("algorithms")) {
        config.algorithms.clear();
        for (const auto &name : data["algorithms"])
            config.algorithms.push_back(parse_enum<selection::AlgorithmType>(name));
    }

    parse_split_config(config, data.value("split", json::object()));
    parse_knn_config(config, data.value("knn", json::object()));
    parse_range_config(config, data.value("range", json::object()));
    parse_where_at_config(config, data.value("where_at", json::object()));
    parse_where_at_scoring_config(config, data.value("where_at_scoring", json::object()));

    return config;
}

} // namespace

Config load_config(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file)
        throw std::invalid_argument("config: cannot open " + path.string());

    try {
        return parse_config(json::parse(file, nullptr, true, /*ignore_comments*/ true));
    } catch (const json::exception &e) {
        // Wrong types and malformed JSON
        throw std::invalid_argument(std::string("config: ") + e.what());
    }
}

} // namespace pipeline

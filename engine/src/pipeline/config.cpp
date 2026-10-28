#include "backend/engine/pipeline/config.hpp"

#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pipeline {

namespace {

using json = nlohmann::json;

void validate_keys(const json &object, std::initializer_list<std::string_view> allowed,
                   std::string_view section) {
    if (!object.is_object())
        throw std::invalid_argument("config: " + std::string(section) + " must be an object");
    for (const auto &[key, value] : object.items()) {
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
            throw std::invalid_argument("config: unknown key '" + std::string(section) + "." + key +
                                        "'");
    }
}

// Parses an enum from its enumerator name, e.g. "RDP"
template <typename E> E parse_enum(const json &value) {
    const auto name = value.get<std::string>();
    const auto parsed = magic_enum::enum_cast<E>(name);
    if (!parsed)
        throw std::invalid_argument("config: unknown value '" + name + "'");
    return *parsed;
}

void parse_split_config(Config &config, const json &split) {
    validate_keys(split, {"type", "query_percentage"}, "split");
    if (split.contains("type"))
        config.split.type = parse_enum<split::SplitType>(split["type"]);
    config.split.query_percentage = split.value("query_percentage", config.split.query_percentage);
}

void parse_knn_config(Config &config, const json &knn) {
    validate_keys(knn, {"k", "query_count", "p"}, "knn");
    config.knn.k = knn.value("k", config.knn.k);
    config.knn.query_count = knn.value("query_count", config.knn.query_count);
    config.knn.p = knn.value("p", config.knn.p);
}

void parse_range_config(Config &config, const json &range) {
    validate_keys(range,
                  {"query_count", "box_size_m", "window_seconds", "min_results", "full_accept_at",
                   "attempts_per_query"},
                  "range");
    config.range.query_count = range.value("query_count", config.range.query_count);
    config.range.box_size_m = range.value("box_size_m", config.range.box_size_m);
    config.range.window_seconds = range.value("window_seconds", config.range.window_seconds);
    config.range.min_results = range.value("min_results", config.range.min_results);
    config.range.full_accept_at = range.value("full_accept_at", config.range.full_accept_at);
    config.range.attempts_per_query =
        range.value("attempts_per_query", config.range.attempts_per_query);
}

void parse_where_at_config(Config &config, const json &where_at) {
    validate_keys(where_at, {"probes_per_trajectory", "max_gap_seconds"}, "where_at");
    config.where_at.probes_per_trajectory =
        where_at.value("probes_per_trajectory", config.where_at.probes_per_trajectory);
    config.where_at.max_gap_seconds =
        where_at.value("max_gap_seconds", config.where_at.max_gap_seconds);
}

void parse_where_at_scoring_config(Config &config, const json &scoring) {
    validate_keys(scoring, {"hit_radius_m", "min_tolerance_m", "max_tolerance_m"},
                  "where_at_scoring");
    config.where_at_scoring.hit_radius_m =
        scoring.value("hit_radius_m", config.where_at_scoring.hit_radius_m);
    config.where_at_scoring.min_tolerance_m =
        scoring.value("min_tolerance_m", config.where_at_scoring.min_tolerance_m);
    config.where_at_scoring.max_tolerance_m =
        scoring.value("max_tolerance_m", config.where_at_scoring.max_tolerance_m);
}

Config parse_config(const json &data) {
    validate_keys(data,
                  {"data_path", "seed", "budget_shares", "algorithms", "split", "knn", "range",
                   "where_at", "where_at_scoring"},
                  "root");
    Config config;

    if (!data.contains("data_path"))
        throw std::invalid_argument("config: data_path is required");
    config.data_path = data["data_path"].get<std::string>();

    // Every other key is optional and falls back to the struct's default
    config.budget_shares = data.value("budget_shares", config.budget_shares);
    for (const double share : config.budget_shares) {
        if (!std::isfinite(share) || share < 0.0 || share > 1.0)
            throw std::invalid_argument("config: budget_shares must be finite and between 0 and 1");
    }
    config.seed = data.value("seed", config.seed);
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

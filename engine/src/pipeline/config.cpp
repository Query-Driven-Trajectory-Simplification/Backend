#include "backend/engine/pipeline/config.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>

namespace pipeline {

namespace {

using nlohmann::json;

// Overwrites value with object[key] if the key is present
template <typename T> void read(const json &object, const char *key, T &value) {
    const auto it = object.find(key);
    if (it == object.end())
        return;
    try {
        it->get_to(value);
    } catch (const json::exception &e) {
        throw std::invalid_argument(std::string("config: bad value for '") + key +
                                    "': " + e.what());
    }
}

} // namespace

} // namespace pipeline

// Found by nlohmann::json through argument-dependent lookup, so they must live in the namespace of
// the struct they fill. A missing key keeps the struct's default.
namespace query {

void from_json(const nlohmann::json &object, KnnConfig &knn) {
    using pipeline::read;
    read(object, "k", knn.k);
    read(object, "query_count", knn.query_count);
    read(object, "seed", knn.seed);
}

void from_json(const nlohmann::json &object, RangeConfig &range) {
    using pipeline::read;
    read(object, "query_count", range.query_count);
    read(object, "box_size_m", range.box_size_m);
    read(object, "window_seconds", range.window_seconds);
    read(object, "min_results", range.min_results);
    read(object, "full_accept_at", range.full_accept_at);
    read(object, "attempts_per_query", range.attempts_per_query);
    read(object, "seed", range.seed);
}

void from_json(const nlohmann::json &object, WhereAtConfig &where_at) {
    using pipeline::read;
    read(object, "probes_per_trajectory", where_at.probes_per_trajectory);
    read(object, "seed", where_at.seed);
    read(object, "max_gap_seconds", where_at.max_gap_seconds);
}

} // namespace query

namespace selection {

void from_json(const nlohmann::json &value, AlgorithmType &algorithm) {
    const auto name = value.get<std::string>();
    if (name == "RDP")
        algorithm = AlgorithmType::RDP;
    else if (name == "SED")
        algorithm = AlgorithmType::SED;
    else
        throw std::invalid_argument("config: unknown algorithm '" + name + "'");
}

} // namespace selection

namespace evaluator {

void from_json(const nlohmann::json &object, WhereAtScoringConfig &scoring) {
    using pipeline::read;
    read(object, "hit_radius_m", scoring.hit_radius_m);
    read(object, "min_tolerance_m", scoring.min_tolerance_m);
    read(object, "max_tolerance_m", scoring.max_tolerance_m);
}

} // namespace evaluator

namespace pipeline {

Config load_config(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file)
        throw std::invalid_argument("config: cannot open " + path.string());
    const json root = json::parse(file);

    Config config;
    std::string data_path;
    read(root, "data_path", data_path);
    if (data_path.empty())
        throw std::invalid_argument("config: data_path is required");
    config.data_path = data_path;

    read(root, "query_percentage", config.query_percentage);
    read(root, "split_seed", config.split_seed);
    read(root, "budget_shares", config.budget_shares);
    read(root, "algorithms", config.algorithms);
    read(root, "knn", config.knn);
    read(root, "range", config.range);
    read(root, "where_at", config.where_at);
    read(root, "where_at_scoring", config.where_at_scoring);

    return config;
}

} // namespace pipeline

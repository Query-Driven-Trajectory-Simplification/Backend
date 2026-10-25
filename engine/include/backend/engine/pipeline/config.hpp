#pragma once

#include "backend/engine/evaluator/where_at.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/where_at_query.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/split/split.hpp"

#include <filesystem>
#include <vector>

namespace pipeline {

struct Config {
    std::filesystem::path data_path;
    split::SplitConfig split;
    std::vector<double> budget_shares{0.0025, 0.005, 0.01, 0.02};
    std::vector<selection::AlgorithmType> algorithms{selection::AlgorithmType::RDP,
                                                     selection::AlgorithmType::SED};
    query::KnnConfig knn;
    query::RangeConfig range;
    query::WhereAtConfig where_at;
    evaluator::WhereAtScoringConfig where_at_scoring;
};

// Reads a JSON config. Only data_path is required, a missing key keeps its default.
Config load_config(const std::filesystem::path &path);

} // namespace pipeline

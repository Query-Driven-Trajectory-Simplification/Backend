#pragma once

#include "backend/engine/evaluator/where_at.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/route_clustering.hpp"
#include "backend/engine/query/where_at_query.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/split/split.hpp"
#include "backend/engine/teacher/objective.hpp"
#include "backend/engine/teacher/teacher.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace pipeline {

struct Config {
    std::filesystem::path data_path;
    bool export_results = false;
    std::filesystem::path export_dir = "output/queries";
    std::uint64_t seed = 42;
    split::SplitConfig split;
    std::vector<double> allocation_alphas{1.0};
    std::vector<double> budget_shares{0.0025, 0.005, 0.01, 0.02};
    std::vector<selection::AlgorithmType> algorithms{selection::AlgorithmType::RDP,
                                                     selection::AlgorithmType::SED};
    std::vector<std::string> external_algorithms;
    std::filesystem::path selection_dir = "data/selection";
    query::KnnConfig knn;
    query::RangeConfig range;
    query::WhereAtConfig where_at;
    evaluator::WhereAtScoringConfig where_at_scoring;
    query::JoinConfig join;
    bool self_join = false; // diagnostic, outside J4
    // Extra thresholds averaged into the join and clustering scores
    std::vector<double> join_score_eps_m;
    query::RouteClusteringConfig clustering;
    teacher::TeacherConfig teacher;
};

// Reads a JSON config. Only data_path is required, a missing key keeps its default.
Config load_config(const std::filesystem::path &path);

teacher::WorkloadConfig teacher_workload_config(const Config &config);

teacher::ObjectiveConfig teacher_objective_config(const Config &config);

} // namespace pipeline

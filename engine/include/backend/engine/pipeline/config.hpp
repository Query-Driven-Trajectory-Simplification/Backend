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
    query::KnnConfig knn;
    query::RangeConfig range;
    query::WhereAtConfig where_at;
    evaluator::WhereAtScoringConfig where_at_scoring;
    query::JoinConfig join;
    bool self_join = false; // also scores the database self join at join.eps_m, outside J4
    query::RouteClusteringConfig clustering;
    teacher::TeacherConfig teacher; // off unless enabled
};

// Reads a JSON config. Only data_path is required, a missing key keeps its default.
Config load_config(const std::filesystem::path &path);

// The teacher's training workload settings: the test's kNN, join, Fréchet and where-at settings,
// with the teacher's own range query count
teacher::WorkloadConfig teacher_workload_config(const Config &config);

// The teacher's objective settings: the where-at scoring ramp and the teacher section's options
teacher::ObjectiveConfig teacher_objective_config(const Config &config);

} // namespace pipeline

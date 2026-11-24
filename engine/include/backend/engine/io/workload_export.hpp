#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/route_clustering.hpp"

#include <filesystem>
#include <span>
#include <string_view>

namespace io {

void export_workloads(const std::filesystem::path &dir, const core::TrajectoryDb &database,
                      std::span<const query::RangeQuery> range, const query::KnnWorkload &knn,
                      const query::JoinWorkload &join,
                      const query::RouteClusteringWorkload *clustering);

// Exports to RL4QDTS
void export_database_ids(const std::filesystem::path &dir, const core::TrajectoryDb &database);

// Exports to RL4QDTS: one row per query with its box (local metres, Unix seconds) and the
// trajectory ids it finds in the database
void export_range_boxes(const std::filesystem::path &path, const core::TrajectoryDb &database,
                        std::span<const query::RangeQuery> queries);

void export_results(const std::filesystem::path &dir, std::string_view name,
                    const evaluator::EvaluationResult &evaluation, bool clustering);

void export_trajectories(const std::filesystem::path &dir, std::string_view name,
                         const core::TrajectoryDb &database);

} // namespace io

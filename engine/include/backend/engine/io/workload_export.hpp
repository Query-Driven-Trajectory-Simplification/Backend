#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"

#include <filesystem>
#include <span>

namespace io {

void export_workloads(const std::filesystem::path &dir, const core::TrajectoryDb &database,
                      std::span<const query::RangeQuery> range, const query::KnnWorkload &knn);

} // namespace io

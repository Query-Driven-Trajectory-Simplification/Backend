#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/query/heat_map.hpp"
#include "backend/engine/query/range_query.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace pipeline {

struct QuerySet {
    query::HeatMap heat_map;
    query::RangeQueryParams params;
    std::vector<query::RangeQuery> range_queries;
};

QuerySet generate_query_set(const core::TrajectoryDb &trajectories, const Config &config,
                            std::size_t count);

void write_query_set(const QuerySet &set, const Config &config,
                     const std::filesystem::path &output_dir);

} // namespace pipeline

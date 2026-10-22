#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/where_at_query.hpp"

#include <limits>
#include <span>

namespace evaluator {

struct WhereAtScores {
    double within_100m = std::numeric_limits<double>::quiet_NaN();
    double average_hit_25_500m = std::numeric_limits<double>::quiet_NaN();
};

// Scores are in [0,1], averaged within trips first, then equally across trips.
// average_hit_25_500m averages success over a UNIFORM tolerance in [25,500]m:
// each error e contributes clamp((500-e)/475, 0, 1).
WhereAtScores evaluate_where_at(const core::TrajectoryDb &compressed,
                                std::span<const query::WhereAtQuery> queries);

} // namespace evaluator

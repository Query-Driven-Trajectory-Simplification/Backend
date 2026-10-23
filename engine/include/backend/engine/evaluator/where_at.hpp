#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/where_at_query.hpp"

#include <limits>
#include <span>

namespace evaluator {

struct WhereAtScoringConfig {
    double hit_radius_m = 100.0;
    double min_tolerance_m = 25.0;
    double max_tolerance_m = 500.0;
};

struct WhereAtScores {
    double within_radius = std::numeric_limits<double>::quiet_NaN();
    double average_hit = std::numeric_limits<double>::quiet_NaN();
};

// Scores are in [0,1], averaged within trips first, then equally across trips.
// average_hit averages success over a uniform tolerance in the configured range.
// Distances must be finite and nonnegative, with min_tolerance_m < max_tolerance_m.
WhereAtScores evaluate_where_at(const core::TrajectoryDb &compressed,
                                std::span<const query::WhereAtQuery> queries,
                                const WhereAtScoringConfig &config = {});

} // namespace evaluator

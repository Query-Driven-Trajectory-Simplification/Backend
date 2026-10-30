#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/range_query.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace evaluator {

struct RangeEvaluation {
    double recall = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::vector<std::int32_t>> answers;
};

RangeEvaluation evaluate_range(const core::TrajectoryDb &compressed,
                               std::span<const query::RangeQuery> queries);

} // namespace evaluator

namespace evaluator::range {

// Share of original that also appears in simplified. NaN if original is empty.
double recall(const std::vector<std::size_t> &original, const std::vector<std::size_t> &simplified);

} // namespace evaluator::range

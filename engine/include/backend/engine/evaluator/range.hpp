#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/range_query.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace evaluator {

// Recall averaged over the queries where it is defined, NaN if it is defined for none.
double evaluate_range(const core::TrajectoryDb &compressed,
                      std::span<const query::RangeQuery> queries);

} // namespace evaluator

namespace evaluator::range {

// Share of original that also appears in simplified. NaN if original is empty.
double recall(const std::vector<std::size_t> &original, const std::vector<std::size_t> &simplified);

} // namespace evaluator::range

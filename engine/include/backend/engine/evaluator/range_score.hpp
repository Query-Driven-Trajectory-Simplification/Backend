#pragma once

#include <cstdint>

namespace evaluator::range {

// How much of what the original returned for one range query the simplified database still
// returns, in [0, 1]. The simplified points are a subset of the original's, so this is recall.
// NaN if the original returned nothing.
double query_score(std::uint32_t original_results, std::uint32_t simplified_results);

} // namespace evaluator::range

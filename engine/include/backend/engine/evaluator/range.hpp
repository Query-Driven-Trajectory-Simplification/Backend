#pragma once

#include <cstddef>
#include <vector>

namespace evaluator::range {

// Share of original that also appears in simplified. NaN if original is empty.
double recall(const std::vector<std::size_t> &original, const std::vector<std::size_t> &simplified);

} // namespace evaluator::range

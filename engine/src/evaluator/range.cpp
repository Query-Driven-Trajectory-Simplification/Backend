#include "backend/engine/evaluator/range.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace evaluator::range {

double recall(const std::vector<std::size_t> &original,
              const std::vector<std::size_t> &simplified) {
    if (original.empty())
        return std::numeric_limits<double>::quiet_NaN();

    // Both come from query::range, so they are sorted
    std::size_t hits = 0;
    for (const std::size_t i : simplified)
        hits += std::ranges::binary_search(original, i);
    return static_cast<double>(hits) / static_cast<double>(original.size());
}

} // namespace evaluator::range

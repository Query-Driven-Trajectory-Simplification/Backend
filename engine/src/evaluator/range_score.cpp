#include "backend/engine/evaluator/range_score.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace evaluator::range {

double query_score(std::uint32_t original_results, std::uint32_t simplified_results) {
    if (original_results == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto error = std::abs(static_cast<double>(original_results) -
                                static_cast<double>(simplified_results)) /
                       static_cast<double>(original_results);
    return std::max(0.0, 1.0 - error);
}

} // namespace evaluator::range

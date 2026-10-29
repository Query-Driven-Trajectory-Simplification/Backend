#include "backend/engine/evaluator/range.hpp"
#include "backend/engine/core/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace evaluator {

RangeEvaluation evaluate_range(const core::TrajectoryDb &compressed,
                               std::span<const query::RangeQuery> queries) {
    RangeEvaluation result;
    result.answers.resize(queries.size());
    double sum = 0.0;
    std::size_t scored = 0;
    std::vector<double> recalls(queries.size());
    core::parallel_for(queries.size(), [&](std::size_t i) {
        const auto &range_query = queries[i];
        const auto simplified = query::range(compressed, range_query.box);
        auto &answer = result.answers[i];
        answer.reserve(simplified.size());
        for (const auto index : simplified)
            answer.push_back(compressed.trajectory(index).trajectory_id);
        recalls[i] = range::recall(range_query.reference, simplified);
    });
    for (const double recall : recalls) {
        if (!std::isnan(recall)) {
            sum += recall;
            ++scored;
        }
    }
    if (scored != 0)
        result.recall = sum / static_cast<double>(scored);
    return result;
}

} // namespace evaluator

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

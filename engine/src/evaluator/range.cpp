#include "backend/engine/evaluator/range.hpp"
#include "backend/engine/core/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace evaluator {

RangeEvaluation evaluate_range(const core::TrajectoryDb &compressed,
                               std::span<const query::RangeQuery> queries) {
    RangeEvaluation result;
    result.answers.resize(queries.size());
    double sum = 0.0;
    std::size_t scored = 0;
    std::vector<double> recalls(queries.size()), f1s(queries.size());
    core::parallel_for(queries.size(), [&](std::size_t i) {
        const auto &range_query = queries[i];
        std::vector<std::size_t> simplified;
        for (const auto index : range_query.reference) {
            if (index >= compressed.size())
                throw std::invalid_argument("range: reference trajectory out of bounds");
            if (query::intersects(compressed.trajectory(index), range_query.box))
                simplified.push_back(index);
        }
        auto &answer = result.answers[i];
        answer.reserve(simplified.size());
        for (const auto index : simplified)
            answer.push_back(compressed.trajectory(index).trajectory_id);
        recalls[i] = range::recall(range_query.reference, simplified);
        f1s[i] = range::f1(range_query.reference, simplified);
    });
    for (const double recall : recalls) {
        if (!std::isnan(recall)) {
            sum += recall;
            ++scored;
        }
    }
    if (scored != 0)
        result.recall = sum / static_cast<double>(scored);
    sum = 0.0;
    scored = 0;
    for (const double f1 : f1s) {
        if (!std::isnan(f1)) {
            sum += f1;
            ++scored;
        }
    }
    if (scored != 0)
        result.f1 = sum / static_cast<double>(scored);
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

double f1(const std::vector<std::size_t> &original, const std::vector<std::size_t> &simplified) {
    if (original.empty() && simplified.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::size_t hits = 0;
    for (const auto i : simplified)
        hits += std::ranges::binary_search(original, i);
    return 2.0 * static_cast<double>(hits) /
           (static_cast<double>(original.size()) + static_cast<double>(simplified.size()));
}

} // namespace evaluator::range

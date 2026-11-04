#include "backend/engine/evaluator/join.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/query/frechet_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace evaluator {

namespace {

struct Counts {
    std::size_t found = 0;
    std::size_t hits = 0;
};

double average(const std::vector<double> &values) {
    double sum = 0.0;
    std::size_t counted = 0;
    for (const double value : values) {
        if (!std::isnan(value)) {
            sum += value;
            ++counted;
        }
    }
    return counted == 0 ? std::numeric_limits<double>::quiet_NaN()
                        : sum / static_cast<double>(counted);
}

} // namespace

JoinScores evaluate_join(const core::TrajectoryDb &compressed,
                         const query::JoinWorkload &workload) {
    const std::size_t n = workload.entries.size();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> f1(n, nan), precision(n, nan), recall(n, nan);

    core::parallel_for(n, [&](std::size_t i) {
        const auto &entry = workload.entries[i];
        const auto query = workload.queries.trajectory(entry.query_index);

        // Simplification keeps endpoints, so the original candidates cover every possible match
        Counts counts;
        for (const auto j : entry.candidates) {
            if (!query::frechet_within(query, compressed.trajectory(j), workload.eps_m))
                continue;
            ++counts.found;
            if (std::ranges::binary_search(entry.reference, j))
                ++counts.hits;
        }

        const auto hits = static_cast<double>(counts.hits);
        const auto found = static_cast<double>(counts.found);
        const auto truth = static_cast<double>(entry.reference.size());
        if (counts.found != 0)
            precision[i] = hits / found;
        if (!entry.reference.empty())
            recall[i] = hits / truth;
        if (counts.found != 0 || !entry.reference.empty())
            f1[i] = 2.0 * hits / (found + truth);
    });

    return {.f1 = average(f1),
            .precision = average(precision),
            .recall = average(recall),
            .eps_m = workload.eps_m,
            .query_count = n};
}

} // namespace evaluator

#include "backend/engine/evaluator/join.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/query/frechet_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
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
    const bool self = workload.mode == query::JoinMode::Self;
    if (self) {
        if (compressed.size() != workload.queries.size() ||
            workload.entries.size() != compressed.size())
            throw std::invalid_argument("self join: database size does not match reference");
        for (std::size_t i = 0; i < compressed.size(); ++i) {
            const auto a = compressed.trajectory(i);
            const auto b = workload.queries.trajectory(i);
            if (a.size() == 0 || b.size() == 0 || a.trajectory_id != b.trajectory_id ||
                a.x.front() != b.x.front() || a.y.front() != b.y.front() ||
                a.x.back() != b.x.back() || a.y.back() != b.y.back())
                throw std::invalid_argument(
                    "self join: IDs/order or endpoints do not match reference");
        }
    }
    const std::size_t n = workload.entries.size();
    std::vector<Counts> totals(n);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> f1, precision, recall;
    if (!self) {
        f1.assign(n, nan);
        precision.assign(n, nan);
        recall.assign(n, nan);
    }

    core::parallel_for(n, [&](std::size_t i) {
        const auto &entry = workload.entries[i];
        const auto query = self ? compressed.trajectory(entry.query_index)
                                : workload.queries.trajectory(entry.query_index);

        // Simplification keeps endpoints, so the original candidates cover every possible match
        Counts counts;
        for (const auto j : entry.candidates) {
            if (!query::frechet_within(query, compressed.trajectory(j), workload.eps_m))
                continue;
            ++counts.found;
            if (std::ranges::binary_search(entry.reference, j))
                ++counts.hits;
        }

        totals[i] = counts;
        if (self)
            return;
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

    JoinScores result{.f1 = self ? nan : average(f1),
                      .precision = self ? nan : average(precision),
                      .recall = self ? nan : average(recall),
                      .eps_m = workload.eps_m,
                      .query_count = self ? 0 : n,
                      .mode = workload.mode};
    for (std::size_t i = 0; i < n; ++i) {
        result.candidate_pairs += workload.entries[i].candidates.size();
        result.reference_pairs += workload.entries[i].reference.size();
        result.predicted_pairs += totals[i].found;
        result.true_positive_pairs += totals[i].hits;
    }
    if (self) {
        const double hits = static_cast<double>(result.true_positive_pairs);
        const double found = static_cast<double>(result.predicted_pairs);
        const double truth = static_cast<double>(result.reference_pairs);
        result.precision = found == 0 ? nan : hits / found;
        result.recall = truth == 0 ? nan : hits / truth;
        result.f1 = found + truth == 0 ? nan : 2.0 * hits / (found + truth);
    }
    return result;
}

} // namespace evaluator

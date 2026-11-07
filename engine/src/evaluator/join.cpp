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
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Simplification keeps endpoints, so the original candidates cover every possible match
    std::vector<std::vector<std::size_t>> matched(n);
    core::parallel_for(n, [&](std::size_t i) {
        const auto &entry = workload.entries[i];
        const auto query = self ? compressed.trajectory(entry.query_index)
                                : workload.queries.trajectory(entry.query_index);
        for (const auto j : entry.candidates) {
            if (query::frechet_within(query, compressed.trajectory(j), workload.eps_m,
                                      workload.frechet))
                matched[i].push_back(j);
        }
    });

    // In Self mode every trajectory is a query, and each pair counts for both of its trajectories
    const std::size_t queries = self ? compressed.size() : n;
    std::vector<Counts> counts(queries);
    std::vector<std::size_t> truth(queries, 0);
    JoinScores result{.eps_m = workload.eps_m, .query_count = queries, .mode = workload.mode};
    for (std::size_t i = 0; i < n; ++i) {
        const auto &entry = workload.entries[i];
        const auto q = self ? entry.query_index : i;
        for (const auto j : matched[i]) {
            const bool hit = std::ranges::binary_search(entry.reference, j);
            ++counts[q].found;
            counts[q].hits += hit;
            if (self) {
                ++counts[j].found;
                counts[j].hits += hit;
            }
            result.true_positive_pairs += hit;
        }
        truth[q] += entry.reference.size();
        if (self) {
            for (const auto j : entry.reference)
                ++truth[j];
        }
        result.candidate_pairs += entry.candidates.size();
        result.reference_pairs += entry.reference.size();
        result.predicted_pairs += matched[i].size();
    }

    std::vector<double> f1(queries, nan), precision(queries, nan), recall(queries, nan);
    for (std::size_t q = 0; q < queries; ++q) {
        const auto hits = static_cast<double>(counts[q].hits);
        const auto found = static_cast<double>(counts[q].found);
        const auto expected = static_cast<double>(truth[q]);
        if (counts[q].found != 0)
            precision[q] = hits / found;
        if (truth[q] != 0)
            recall[q] = hits / expected;
        if (counts[q].found != 0 || truth[q] != 0)
            f1[q] = 2.0 * hits / (found + expected);
    }
    result.f1 = average(f1);
    result.precision = average(precision);
    result.recall = average(recall);

    const auto hits = static_cast<double>(result.true_positive_pairs);
    const auto found = static_cast<double>(result.predicted_pairs);
    const auto expected = static_cast<double>(result.reference_pairs);
    result.micro_precision = found == 0 ? nan : hits / found;
    result.micro_recall = expected == 0 ? nan : hits / expected;
    result.micro_f1 = found + expected == 0 ? nan : 2.0 * hits / (found + expected);
    return result;
}

} // namespace evaluator

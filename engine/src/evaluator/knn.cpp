#include "backend/engine/evaluator/knn.hpp"
#include "backend/engine/core/parallel.hpp"
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>
namespace evaluator::knn {
namespace {
std::vector<std::pair<std::size_t, std::size_t>>
shared_ranks(const std::vector<std::size_t> &original, const std::vector<std::size_t> &simplified) {

    std::vector<std::pair<std::size_t, std::size_t>> shared;

    for (std::size_t i = 0; i < original.size(); i++) {
        for (std::size_t j = 0; j < simplified.size(); j++) {
            if (original[i] == simplified[j]) {
                shared.emplace_back(i, j);
                break;
            }
        }
    }
    return shared;
}
} // namespace

double kendalls_tau(const std::vector<std::size_t> &original,
                    const std::vector<std::size_t> &simplified) {
    std::vector<std::pair<std::size_t, std::size_t>> overlap = shared_ranks(original, simplified);
    if (overlap.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::size_t concordant = 0, discordant = 0;

    for (std::size_t i = 0; i < overlap.size(); i++) {
        for (std::size_t j = i + 1; j < overlap.size(); j++) {
            if (overlap[i].second < overlap[j].second) {
                concordant++;
            } else {
                discordant++;
            }
        }
    }
    return (static_cast<double>(concordant) - static_cast<double>(discordant)) /
           static_cast<double>(concordant + discordant);
}

double recall_at_k(const std::vector<std::size_t> &original,
                   const std::vector<std::size_t> &simplified) {
    if (original.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(shared_ranks(original, simplified).size()) /
           static_cast<double>(original.size());
}

} // namespace evaluator::knn

namespace evaluator {

KnnScores evaluate_knn(const core::TrajectoryDb &compressed, const query::KnnWorkload &workload) {
    double tau_sum = 0.0;
    std::size_t tau_scored = 0;
    double recall_sum = 0.0;
    std::size_t recall_scored = 0;
    std::vector<double> taus(workload.entries.size());
    std::vector<double> recalls(workload.entries.size());
    core::parallel_for(workload.entries.size(), [&](std::size_t i) {
        const auto &query = workload.entries[i];
        const auto trajectory = workload.queries.trajectory(query.query_index);
        const auto simplified = query::knn(compressed, trajectory, workload.k);
        taus[i] = knn::kendalls_tau(query.reference, simplified);
        recalls[i] = knn::recall_at_k(query.reference, simplified);
    });
    for (std::size_t i = 0; i < workload.entries.size(); ++i) {
        const double tau = taus[i];
        if (!std::isnan(tau)) {
            tau_sum += tau;
            ++tau_scored;
        }
        const double recall = recalls[i];
        if (!std::isnan(recall)) {
            recall_sum += recall;
            ++recall_scored;
        }
    }

    KnnScores scores;
    if (tau_scored != 0)
        scores.kendalls_tau = tau_sum / static_cast<double>(tau_scored);
    if (recall_scored != 0)
        scores.recall_at_k = recall_sum / static_cast<double>(recall_scored);
    return scores;
}

} // namespace evaluator

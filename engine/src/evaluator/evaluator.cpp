#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/kendalls_tau.hpp"
#include "backend/engine/query/knn.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace evaluator {

namespace {

// Mean Kendall's tau between the knn results on both databases, over every query.
// Queries whose results share fewer than two trajectories have no tau and are skipped.
double knn_score(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified,
                 const core::TrajectoryDb &queries, std::size_t k) {
    double sum = 0.0;
    std::size_t scored = 0;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto trajectory = queries.trajectory(i);
        const double tau = knn::evaluate_knn(query::knn(original, trajectory, k),
                                             query::knn(simplified, trajectory, k));
        if (!std::isnan(tau)) {
            sum += tau;
            ++scored;
        }
    }
    return scored == 0 ? std::numeric_limits<double>::quiet_NaN()
                       : sum / static_cast<double>(scored);
}

} // namespace

Scoretable evaluate_simplification(const core::TrajectoryDb &original,
                                   const core::TrajectoryDb &simplified,
                                   const core::TrajectoryDb &queries, std::size_t k) {
    if (original.size() != simplified.size())
        throw std::invalid_argument(
            "evaluate_simplification: original and simplified hold different trajectories");

    return Scoretable{.knn_score = knn_score(original, simplified, queries, k)};
}

} // namespace evaluator

#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/kendalls_tau.hpp"
#include "backend/engine/query/knn.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace evaluator {

Evaluator::Evaluator(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified,
                     Workloads workloads)
    : m_original(original), m_simplified(simplified), m_workloads(workloads) {
    if (original.size() != simplified.size())
        throw std::invalid_argument(
            "Evaluator: original and simplified hold different trajectories");
    for (std::size_t i = 0; i < original.size(); ++i) {
        if (original.trajectory(i).trajectory_id != simplified.trajectory(i).trajectory_id)
            throw std::invalid_argument("Evaluator: trajectory IDs/order do not match");
    }
}

Scoretable Evaluator::evaluate() {
    return Scoretable{
        .knn_score = get_knn_score(),
        .range_score = get_range_score(),
        .where_at = get_where_at_scores(),
    };
}

double Evaluator::get_knn_score() {
    if (m_knn_neighbours == 0) {
        m_knn_score = std::numeric_limits<double>::quiet_NaN();
        return m_knn_score;
    }
    knn_score(m_knn_neighbours);
    return m_knn_score;
}

double Evaluator::get_range_score() {
    range_score();
    return m_range_score;
}

WhereAtScores Evaluator::get_where_at_scores() {
    m_where_at_scores = evaluate_where_at(m_simplified, m_workloads.where_at);
    return m_where_at_scores;
}

void Evaluator::set_knn_neighbours(std::size_t k) { m_knn_neighbours = k; }

void Evaluator::knn_score(std::size_t k) {
    const core::TrajectoryDb queries = query::generate_knn_queries(m_original);

    double sum = 0.0;
    std::size_t scored = 0;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto trajectory = queries.trajectory(i);
        const double tau = knn::kendalls_tau(query::knn(m_original, trajectory, k),
                                             query::knn(m_simplified, trajectory, k));
        if (!std::isnan(tau)) {
            sum += tau;
            ++scored;
        }
    }
    m_knn_score =
        scored == 0 ? std::numeric_limits<double>::quiet_NaN() : sum / static_cast<double>(scored);
}

void Evaluator::range_score() { m_range_score = std::numeric_limits<double>::quiet_NaN(); }

} // namespace evaluator

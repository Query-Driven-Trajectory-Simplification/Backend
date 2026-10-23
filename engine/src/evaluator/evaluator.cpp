#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/knn_evaluation.hpp"
#include "backend/engine/query/knn.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace evaluator {

Evaluator::Evaluator(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified,
                     Workloads workloads, WhereAtScoringConfig where_at_config)
    : m_original(original), m_simplified(simplified), m_workloads(workloads),
      m_where_at_config(where_at_config) {
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
        .knn_kendalls_tau = get_knn_kendalls_tau(),
        .knn_recall_at_k = get_knn_recall_at_k(),
        .range_score = get_range_score(),
        .where_at = get_where_at_scores(),
    };
}

double Evaluator::get_knn_kendalls_tau() {
    knn_scores();
    return m_knn_kendalls_tau;
}

double Evaluator::get_knn_recall_at_k() {
    knn_scores();
    return m_knn_recall_at_k;
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
    m_where_at_scores = evaluate_where_at(m_simplified, m_workloads.where_at, m_where_at_config);
    return m_where_at_scores;
}

void Evaluator::set_knn_workload(const query::KnnWorkload &workload) { m_knn_workload = &workload; }

void Evaluator::knn_scores() {
    m_knn_kendalls_tau = std::numeric_limits<double>::quiet_NaN();
    m_knn_recall_at_k = std::numeric_limits<double>::quiet_NaN();
    if (m_knn_workload == nullptr) {
        return;
    }
    const auto &workload = *m_knn_workload;

    double tau_sum = 0.0;
    std::size_t tau_scored = 0;
    double recall_sum = 0.0;
    std::size_t recall_scored = 0;
    for (const auto &query : workload.entries) {
        const auto trajectory = workload.queries.trajectory(query.query_index);
        const auto simplified = query::knn(m_simplified, trajectory, workload.k);

        const double tau = knn::kendalls_tau(query.reference, simplified);
        if (!std::isnan(tau)) {
            tau_sum += tau;
            ++tau_scored;
        }
        const double recall = knn::recall_at_k(query.reference, simplified);
        if (!std::isnan(recall)) {
            recall_sum += recall;
            ++recall_scored;
        }
    }
    if (tau_scored != 0)
        m_knn_kendalls_tau = tau_sum / static_cast<double>(tau_scored);
    if (recall_scored != 0)
        m_knn_recall_at_k = recall_sum / static_cast<double>(recall_scored);
    m_knn_score =
        scored == 0 ? std::numeric_limits<double>::quiet_NaN() : sum / static_cast<double>(scored);
}

void Evaluator::range_score() { m_range_score = std::numeric_limits<double>::quiet_NaN(); }

void Evaluator::where_at_score() { m_where_at_score = std::numeric_limits<double>::quiet_NaN(); }

} // namespace evaluator

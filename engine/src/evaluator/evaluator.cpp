#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/evaluator/knn_evaluation.hpp"
#include "backend/engine/evaluator/range.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace evaluator {

Evaluator::Evaluator(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified,
                     Workloads workloads, WhereAtScoringConfig where_at_config)
    : m_simplified(simplified), m_workloads(workloads), m_where_at_config(where_at_config) {
    if (original.size() != simplified.size())
        throw std::invalid_argument(
            "Evaluator: original and simplified hold different trajectories");
    for (std::size_t i = 0; i < original.size(); ++i) {
        if (original.trajectory(i).trajectory_id != simplified.trajectory(i).trajectory_id)
            throw std::invalid_argument("Evaluator: trajectory IDs/order do not match");
    }
}

Scoretable Evaluator::evaluate() {
    knn_scores();
    return Scoretable{
        .knn_kendalls_tau = m_knn_kendalls_tau,
        .knn_recall_at_k = m_knn_recall_at_k,
        .range_recall = get_range_recall(),
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
}

double Evaluator::get_range_recall() {
    range_recall();
    return m_range_recall;
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

    std::vector<double> taus(workload.entries.size());
    std::vector<double> recalls(workload.entries.size());
    core::parallel_for(workload.entries.size(), [&](std::size_t i) {
        const auto &query = workload.entries[i];
        const auto trajectory = workload.queries.trajectory(query.query_index);
        const auto simplified = query::knn(m_simplified, trajectory, workload.k);
        taus[i] = knn::kendalls_tau(query.reference, simplified);
        recalls[i] = knn::recall_at_k(query.reference, simplified);
    });

    double tau_sum = 0.0;
    std::size_t tau_scored = 0;
    double recall_sum = 0.0;
    std::size_t recall_scored = 0;
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
    if (tau_scored != 0)
        m_knn_kendalls_tau = tau_sum / static_cast<double>(tau_scored);
    if (recall_scored != 0)
        m_knn_recall_at_k = recall_sum / static_cast<double>(recall_scored);
}

void Evaluator::range_recall() {
    m_range_recall = std::numeric_limits<double>::quiet_NaN();

    double sum = 0.0;
    std::size_t scored = 0;
    for (const auto &range_query : m_workloads.range) {
        const auto simplified = query::range(m_simplified, range_query.box);
        const double recall = range::recall(range_query.reference, simplified);
        if (!std::isnan(recall)) {
            sum += recall;
            ++scored;
        }
    }
    if (scored != 0)
        m_range_recall = sum / static_cast<double>(scored);
}

} // namespace evaluator

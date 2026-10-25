#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/knn_evaluation.hpp"
#include "backend/engine/evaluator/range.hpp"

#include <cstddef>
#include <stdexcept>

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
    return Scoretable{
        .knn = get_knn_scores(),
        .range_recall = get_range_recall(),
        .where_at = get_where_at_scores(),
    };
}

double Evaluator::get_range_recall() {
    m_range_recall = evaluate_range(m_simplified, m_workloads.range);
    return m_range_recall;
}

WhereAtScores Evaluator::get_where_at_scores() {
    m_where_at_scores = evaluate_where_at(m_simplified, m_workloads.where_at, m_where_at_config);
    return m_where_at_scores;
}

KnnScores Evaluator::get_knn_scores() {
    m_knn_scores = m_workloads.knn ? evaluate_knn(m_simplified, *m_workloads.knn) : KnnScores{};
    return m_knn_scores;
}

} // namespace evaluator

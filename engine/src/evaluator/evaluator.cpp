#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/knn.hpp"
#include "backend/engine/evaluator/range.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

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

EvaluationResult Evaluator::evaluate() const {
    auto knn = m_workloads.knn ? evaluate_knn(m_simplified, *m_workloads.knn) : KnnEvaluation{};
    auto range = evaluate_range(m_simplified, m_workloads.range);
    return {.scores = {.knn = knn.scores,
                       .range_recall = range.recall,
                       .where_at = get_where_at_scores()},
            .knn = std::move(knn.answers),
            .range = std::move(range.answers)};
}

KnnScores Evaluator::get_knn_scores() const {
    return m_workloads.knn ? evaluate_knn(m_simplified, *m_workloads.knn).scores : KnnScores{};
}

double Evaluator::get_range_recall() const {
    return evaluate_range(m_simplified, m_workloads.range).recall;
}

WhereAtScores Evaluator::get_where_at_scores() const {
    return evaluate_where_at(m_simplified, m_workloads.where_at, m_where_at_config);
}

} // namespace evaluator

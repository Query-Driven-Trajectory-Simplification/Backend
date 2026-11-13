#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/knn.hpp"
#include "backend/engine/evaluator/range.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace evaluator {

namespace {

bool can_reuse_join(const core::TrajectoryDb &db, const query::JoinWorkload &join,
                    const query::RouteClusteringWorkload &clustering) {
    if (join.mode != query::JoinMode::Self || join.eps_m != clustering.eps_m ||
        join.frechet != clustering.frechet || clustering.originals.size() != db.size())
        return false;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto trajectory = db.trajectory(i);
        if (trajectory.size() == 0 || query::endpoints_of(trajectory) != clustering.originals[i])
            return false;
    }
    // Also require the same candidate pairs: independently supplied workloads may differ.
    std::size_t k = 0;
    for (const auto &entry : join.entries) {
        for (const auto j : entry.candidates) {
            if (k == clustering.candidates.size() ||
                clustering.candidates[k++] != query::TrajectoryPair{entry.query_index, j})
                return false;
        }
    }
    return k == clustering.candidates.size();
}

} // namespace

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
    const bool reuse = m_workloads.join && m_workloads.clustering &&
                       can_reuse_join(m_simplified, *m_workloads.join, *m_workloads.clustering);
    std::vector<query::TrajectoryPair> pairs;
    auto join = m_workloads.join
                    ? run_join(m_simplified, *m_workloads.join, reuse ? &pairs : nullptr)
                    : JoinEvaluation{};

    RouteClusteringScores clustering;
    std::vector<std::vector<std::int32_t>> clusters;
    if (m_workloads.clustering) {
        const auto predicted =
            reuse ? query::cluster_routes(m_simplified, pairs, m_workloads.clustering->min_size)
                  : compressed_route_clusters(m_simplified, *m_workloads.clustering);
        clustering = compare_route_clusters(m_workloads.clustering->reference, predicted);
        clusters.reserve(predicted.members.size());
        for (const auto &members : predicted.members) {
            auto &ids = clusters.emplace_back();
            ids.reserve(members.size());
            for (const auto i : members)
                ids.push_back(m_simplified.trajectory(i).trajectory_id);
        }
    }

    return {.scores = {.knn = knn.scores,
                       .range_recall = range.recall,
                       .where_at = get_where_at_scores(),
                       .join = join.scores,
                       .range_f1 = range.f1,
                       .clustering = clustering},
            .knn = std::move(knn.answers),
            .range = std::move(range.answers),
            .join = std::move(join.pairs),
            .clusters = std::move(clusters)};
}

KnnScores Evaluator::get_knn_scores() const {
    return m_workloads.knn ? evaluate_knn(m_simplified, *m_workloads.knn).scores : KnnScores{};
}

double Evaluator::get_range_recall() const {
    return evaluate_range(m_simplified, m_workloads.range).recall;
}

double Evaluator::get_range_f1() const {
    return evaluate_range(m_simplified, m_workloads.range).f1;
}

WhereAtScores Evaluator::get_where_at_scores() const {
    return evaluate_where_at(m_simplified, m_workloads.where_at, m_where_at_config);
}

JoinScores Evaluator::get_join_scores() const {
    return m_workloads.join ? evaluate_join(m_simplified, *m_workloads.join) : JoinScores{};
}

RouteClusteringScores Evaluator::get_clustering_scores() const {
    return m_workloads.clustering ? evaluate_route_clustering(m_simplified, *m_workloads.clustering)
                                  : RouteClusteringScores{};
}

} // namespace evaluator

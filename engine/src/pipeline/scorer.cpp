#include "backend/engine/pipeline/scorer.hpp"
#include "backend/engine/evaluator/combined.hpp"
#include "backend/engine/io/workload_export.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace pipeline {

namespace {

void print_scores(const evaluator::Scoretable &scores) {
    std::cout << "  knn top-k tau: " << scores.knn.kendalls_tau_topk
              << ", recall@k: " << scores.knn.recall_at_k << " (" << scores.knn.query_count
              << " queries)\n";
    std::cout << "  where-at: average hit=" << scores.where_at.average_hit
              << ", within radius=" << scores.where_at.within_radius << '\n';
    std::cout << "  range F1: " << scores.range_f1 << ", recall: " << scores.range_recall << '\n';
    std::cout << "  join F1: " << scores.join.f1 << ", precision: " << scores.join.precision
              << ", recall: " << scores.join.recall << ", micro F1: " << scores.join.micro_f1
              << '\n';
    if (scores.join.mode != query::JoinMode::Self && scores.self_join.query_count != 0)
        std::cout << "  self join F1: " << scores.self_join.f1
                  << ", precision: " << scores.self_join.precision
                  << ", recall: " << scores.self_join.recall << '\n';
    if (!std::isnan(scores.clustering.score) || scores.clustering.reference_clusters != 0)
        std::cout << "  route clustering: " << scores.clustering.score << " ("
                  << scores.clustering.predicted_clusters << " clusters, "
                  << scores.clustering.predicted_covered << " trajectories)\n";
}

void print_normalized(const evaluator::NormalizedScores &scores) {
    std::cout << "  normalized: knn=" << scores.knn << ", range=" << scores.range
              << ", where-at=" << scores.where_at << ", join=" << scores.join;
    if (!std::isnan(scores.clustering))
        std::cout << ", clustering=" << scores.clustering;
    if (!std::isnan(scores.self_join))
        std::cout << ", self join=" << scores.self_join;
    std::cout << ", combined J4=" << scores.combined << '\n';
}

core::Simplification endpoints_only(const core::TrajectoryDb &trajectories) {
    core::Simplification simplification;
    for (std::size_t i = 0; i < trajectories.size(); ++i) {
        const std::size_t n = trajectories.trajectory(i).size();
        std::vector<std::size_t> kept;
        if (n > 0)
            kept.push_back(0);
        if (n > 1)
            kept.push_back(n - 1);
        simplification.add_trajectory(kept);
    }
    return simplification;
}

} // namespace

void print_size(std::string_view name, const core::TrajectoryDb &trajectories) {
    std::cout << name << ": " << trajectories.size() << " trajectories, "
              << trajectories.num_points() << " points\n";
}

Scorer::Scorer(const core::TrajectoryDb &database, const TestWorkloads &workloads,
               const Config &config)
    : m_database(database), m_workloads(workloads.view()), m_config(config) {
    const auto floor_db = core::materialize(database, endpoints_only(database));
    std::cout << "floor, endpoints only: ";
    print_size("compressed", floor_db);
    m_floor = evaluator::Evaluator(database, floor_db, m_workloads, config.where_at_scoring)
                  .evaluate()
                  .scores;
    print_scores(m_floor);
}

void Scorer::score(std::string_view label, std::string_view name,
                   const core::Simplification &simplification) const {
    const auto compressed = core::materialize(m_database, simplification);
    std::cout << label << ": ";
    print_size("compressed", compressed);
    const auto evaluation =
        evaluator::Evaluator(m_database, compressed, m_workloads, m_config.where_at_scoring)
            .evaluate();
    print_scores(evaluation.scores);
    print_normalized(evaluator::normalize_scores(evaluation.scores, m_floor));
    if (m_config.export_results) {
        io::export_trajectories(m_config.export_dir, name, compressed);
        io::export_results(m_config.export_dir, name, evaluation);
    }
}

} // namespace pipeline

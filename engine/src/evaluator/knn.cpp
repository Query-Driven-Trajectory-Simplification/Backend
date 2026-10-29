#include "backend/engine/evaluator/knn.hpp"
#include "backend/engine/core/parallel.hpp"
#include <algorithm>
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

std::vector<std::size_t> positions_in(const std::vector<std::size_t> &ranking,
                                      const std::vector<std::size_t> &full) {
    const std::size_t absent = ranking.size();
    std::vector<std::size_t> positions(full.size(), absent);
    for (std::size_t i = 0; i < full.size(); i++) {
        for (std::size_t j = 0; j < ranking.size(); j++) {
            if (ranking[j] == full[i]) {
                positions[i] = j;
                break;
            }
        }
    }
    return positions;
}

} // namespace

// Fagin et al., "Comparing top k lists" (2003)
double kendalls_tau_topk(const std::vector<std::size_t> &original,
                         const std::vector<std::size_t> &simplified, double p) {

    // the lists are in rank order, not sorted, so std::set_union does not apply
    std::vector<std::size_t> union_vec = original;
    for (std::size_t id : simplified) {
        if (std::find(original.begin(), original.end(), id) == original.end()) {
            union_vec.push_back(id);
        }
    }

    const auto original_rank = positions_in(original, union_vec);
    const auto simplified_rank = positions_in(simplified, union_vec);

    double distance = 0.0;
    for (std::size_t i = 0; i < union_vec.size(); i++) {
        for (std::size_t j = i + 1; j < union_vec.size(); j++) {
            if (original_rank[i] == original_rank[j] || simplified_rank[i] == simplified_rank[j]) {
                distance += p; // one list holds neither item, so it cannot tell their order
            } else if ((original_rank[i] < original_rank[j]) !=
                       (simplified_rank[i] < simplified_rank[j])) {
                distance += 1.0; // the lists order the pair in opposite directions
            }
        }
    }

    const double original_size = static_cast<double>(original.size());
    const double simplified_size = static_cast<double>(simplified.size());
    const double max_distance =
        original_size * simplified_size +
        p * (original_size * (original_size - 1) / 2 + simplified_size * (simplified_size - 1) / 2);
    return 1.0 - distance / max_distance;
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

KnnEvaluation evaluate_knn(const core::TrajectoryDb &compressed,
                           const query::KnnWorkload &workload) {
    KnnEvaluation result;
    result.answers.resize(workload.entries.size());
    double recall_sum = 0.0;
    std::size_t recall_scored = 0;
    double topk_tau_sum = 0.0;
    std::size_t topk_tau_scored = 0;
    std::vector<double> recalls(workload.entries.size());
    std::vector<double> topk_taus(workload.entries.size());
    core::parallel_for(workload.entries.size(), [&](std::size_t i) {
        const auto &query = workload.entries[i];
        const auto trajectory = workload.queries.trajectory(query.query_index);
        const auto simplified = query::knn(compressed, trajectory, workload.k);
        auto &answer = result.answers[i];
        answer.query_trajectory_id = trajectory.trajectory_id;
        answer.neighbors.reserve(simplified.size());
        for (const auto index : simplified)
            answer.neighbors.push_back(compressed.trajectory(index).trajectory_id);
        recalls[i] = knn::recall_at_k(query.reference, simplified);
        topk_taus[i] = knn::kendalls_tau_topk(query.reference, simplified, workload.p);
    });
    for (std::size_t i = 0; i < workload.entries.size(); ++i) {
        const double topk_tau = topk_taus[i];
        if (!std::isnan(topk_tau)) {
            topk_tau_sum += topk_tau;
            ++topk_tau_scored;
        }
        const double recall = recalls[i];
        if (!std::isnan(recall)) {
            recall_sum += recall;
            ++recall_scored;
        }
    }

    auto &scores = result.scores;
    scores.query_count = workload.entries.size();
    if (recall_scored != 0)
        scores.recall_at_k = recall_sum / static_cast<double>(recall_scored);
    if (topk_tau_scored != 0)
        scores.kendalls_tau_topk = topk_tau_sum / static_cast<double>(topk_tau_scored);
    return result;
}

} // namespace evaluator

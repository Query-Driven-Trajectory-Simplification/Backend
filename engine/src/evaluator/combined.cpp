#include "backend/engine/evaluator/combined.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace evaluator {

namespace {

double mean_ignoring_nan(std::span<const double> values) {
    double sum = 0.0;
    std::size_t count = 0;
    for (const double value : values) {
        if (!std::isnan(value)) {
            sum += value;
            ++count;
        }
    }
    return count == 0 ? std::numeric_limits<double>::quiet_NaN() : sum / static_cast<double>(count);
}

// The main normalised score averaged with each grid entry, normalised against its own floor,
// skipping NaNs. raw(entry) reads an entry's unnormalised score
template <typename Entry, typename Raw>
double grid_mean(double main, const std::vector<Entry> &scores, const std::vector<Entry> &floors,
                 Raw raw, std::string_view grid) {
    if (scores.size() != floors.size())
        throw std::invalid_argument("normalize_scores: scores and floor hold different " +
                                    std::string(grid) + " grids");
    std::vector<double> values{main};
    for (std::size_t k = 0; k < scores.size(); ++k)
        values.push_back(normalize(raw(scores[k]), raw(floors[k])));
    return mean_ignoring_nan(values);
}

} // namespace

double normalize(double score, double floor) {
    if (std::isnan(score) || std::isnan(floor) || floor >= 1.0)
        return std::numeric_limits<double>::quiet_NaN();
    return (score - floor) / (1.0 - floor);
}

NormalizedScores normalize_scores(const Scoretable &scores, const Scoretable &floor) {
    NormalizedScores result;
    result.knn_recall_at_k = normalize(scores.knn.recall_at_k, floor.knn.recall_at_k);
    result.knn_kendalls_tau_topk =
        normalize(scores.knn.kendalls_tau_topk, floor.knn.kendalls_tau_topk);
    result.range_recall = normalize(scores.range_recall, floor.range_recall);
    result.range_f1 = normalize(scores.range_f1, floor.range_f1);
    result.where_at_within_radius =
        normalize(scores.where_at.within_radius, floor.where_at.within_radius);
    result.where_at_average_hit =
        normalize(scores.where_at.average_hit, floor.where_at.average_hit);
    result.join_f1 = normalize(scores.join.f1, floor.join.f1);
    result.clustering_score = normalize(scores.clustering.score, floor.clustering.score);

    result.knn = result.knn_recall_at_k;
    result.range = result.range_f1;
    result.where_at = result.where_at_average_hit;
    result.join = grid_mean(
        result.join_f1, scores.join_grid, floor.join_grid,
        [](const JoinScores &join) { return join.f1; }, "join");
    result.clustering = grid_mean(
        result.clustering_score, scores.clustering_grid, floor.clustering_grid,
        [](const RouteClusteringScores &clustering) { return clustering.score; }, "clustering");
    result.self_join = normalize(scores.self_join.f1, floor.self_join.f1);
    result.self_knn = normalize(scores.self_knn.recall_at_k, floor.self_knn.recall_at_k);
    result.self_join_mean = grid_mean(
        result.self_join, scores.self_join_grid, floor.self_join_grid,
        [](const JoinScores &join) { return join.f1; }, "self join");

    // Clustering and the self join stay out (see combined.hpp)
    result.combined =
        mean_ignoring_nan(std::array{result.knn, result.range, result.where_at, result.join});
    const std::array internal{result.self_knn, result.range, result.where_at,
                              result.self_join_mean};
    if (std::ranges::none_of(internal, [](double v) { return std::isnan(v); }))
        result.combined_self = mean_ignoring_nan(internal);
    return result;
}

} // namespace evaluator

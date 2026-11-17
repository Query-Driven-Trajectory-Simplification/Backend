#include "backend/engine/evaluator/combined.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace evaluator {

namespace {

double mean_ignoring_nan(std::initializer_list<double> values) {
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
    if (scores.join_grid.size() != floor.join_grid.size())
        throw std::invalid_argument("normalize_scores: scores and floor hold different join grids");
    double join_sum = result.join_f1;
    std::size_t join_count = std::isnan(result.join_f1) ? 0 : 1;
    if (join_count == 0)
        join_sum = 0.0;
    for (std::size_t k = 0; k < scores.join_grid.size(); ++k) {
        const double v = normalize(scores.join_grid[k].f1, floor.join_grid[k].f1);
        if (!std::isnan(v)) {
            join_sum += v;
            ++join_count;
        }
    }
    result.join = join_count ? join_sum / static_cast<double>(join_count)
                             : std::numeric_limits<double>::quiet_NaN();
    if (scores.clustering_grid.size() != floor.clustering_grid.size())
        throw std::invalid_argument(
            "normalize_scores: scores and floor hold different clustering grids");
    {
        double sum = std::isnan(result.clustering_score) ? 0.0 : result.clustering_score;
        std::size_t count = std::isnan(result.clustering_score) ? 0 : 1;
        for (std::size_t k = 0; k < scores.clustering_grid.size(); ++k) {
            const double v =
                normalize(scores.clustering_grid[k].score, floor.clustering_grid[k].score);
            if (!std::isnan(v)) {
                sum += v;
                ++count;
            }
        }
        result.clustering =
            count ? sum / static_cast<double>(count) : std::numeric_limits<double>::quiet_NaN();
    }
    result.self_join = normalize(scores.self_join.f1, floor.self_join.f1);

    // Clustering and the self join stay out (see combined.hpp)
    result.combined = mean_ignoring_nan({result.knn, result.range, result.where_at, result.join});
    return result;
}

} // namespace evaluator

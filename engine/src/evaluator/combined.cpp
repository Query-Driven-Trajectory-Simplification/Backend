#include "backend/engine/evaluator/combined.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>

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
    result.join = result.join_f1;
    result.clustering = result.clustering_score;

    // Clustering stays out: it repeats the join's neighbour test (see combined.hpp)
    result.combined = mean_ignoring_nan({result.knn, result.range, result.where_at, result.join});
    return result;
}

} // namespace evaluator

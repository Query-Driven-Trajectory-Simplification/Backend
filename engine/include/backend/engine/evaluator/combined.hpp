#pragma once

#include "backend/engine/evaluator/evaluator.hpp"

#include <limits>

namespace evaluator {

// 0 = the floor (endpoints only), 1 = perfect, without clipping. Combined averages
// one primary score per type: recall@k, range F1, where-at average-hit, join F1 and route
// clustering. Missing primaries are skipped, never replaced with a diagnostic score.
// Average-hit is the hit rate averaged over a tolerance drawn uniformly from
// [min_tolerance_m, max_tolerance_m]. The within-radius hit rate at hit_radius_m is a
// diagnostic and does not contribute to the combined score.
struct NormalizedScores {
    static constexpr double missing = std::numeric_limits<double>::quiet_NaN();

    double knn_recall_at_k = missing;
    double knn_kendalls_tau_topk = missing;
    double range_recall = missing;
    double range_f1 = missing;
    double where_at_within_radius = missing;
    double where_at_average_hit = missing;
    double join_f1 = missing;
    double clustering_score = missing;

    double knn = missing;
    double range = missing;
    double where_at = missing;
    double join = missing;
    double clustering = missing;

    double combined = missing;
};

// NaN if an input is NaN or the floor is already 1
double normalize(double score, double floor);

NormalizedScores normalize_scores(const Scoretable &scores, const Scoretable &floor);

} // namespace evaluator

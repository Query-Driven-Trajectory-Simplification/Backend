#pragma once

#include "backend/engine/evaluator/evaluator.hpp"

#include <limits>

namespace evaluator {

// 0 = the floor (endpoints only), 1 = perfect, without clipping. Combined averages
// one primary score per type: recall@k, range F1, where-at within-radius hit rate, join F1.
// Missing primaries are skipped, never replaced with a diagnostic score.
// The radius is configured by hit_radius_m (100 m for the research protocol).
// Average-hit remains a diagnostic and does not contribute to the combined score.
struct NormalizedScores {
    static constexpr double missing = std::numeric_limits<double>::quiet_NaN();

    double knn_recall_at_k = missing;
    double knn_kendalls_tau_topk = missing;
    double range_recall = missing;
    double range_f1 = missing;
    double where_at_within_radius = missing;
    double where_at_average_hit = missing;
    double join_f1 = missing;

    double knn = missing;
    double range = missing;
    double where_at = missing;
    double join = missing;

    double combined = missing;
};

// NaN if an input is NaN or the floor is already 1
double normalize(double score, double floor);

NormalizedScores normalize_scores(const Scoretable &scores, const Scoretable &floor);

} // namespace evaluator

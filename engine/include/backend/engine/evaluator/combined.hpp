#pragma once

#include "backend/engine/evaluator/evaluator.hpp"

#include <limits>

namespace evaluator {

// 0 = the floor (endpoints only), 1 = perfect, unclipped. Combined (J4) averages recall@k,
// range F1, where-at average-hit and join F1, skipping missing ones. Within-radius, clustering
// (it repeats the join's Fréchet test) and the self join (seen by anything fitted to the
// database) are diagnostics only.
// join and clustering average over their threshold grids, each against its own floor;
// join_f1 and clustering_score keep the main threshold alone.
// combined_self (J4-self) scores the in-database setting: self kNN, the same range and where-at,
// and the self join over its grid (self_join_mean). NaN unless all four are present.
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
    double self_join = missing;
    double self_knn = missing;
    double self_join_mean = missing;

    double combined = missing;
    double combined_self = missing;
};

// NaN if an input is NaN or the floor is already 1
double normalize(double score, double floor);

NormalizedScores normalize_scores(const Scoretable &scores, const Scoretable &floor);

} // namespace evaluator

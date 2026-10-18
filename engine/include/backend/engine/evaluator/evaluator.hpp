#pragma once

namespace evaluator {

struct Scoretable {
    double knn_score;
    // add where at, range score, etc..
};

Scoretable evaluate_simplification(void /*Original, simplified, datastructure tbd*/);
} // namespace evaluator
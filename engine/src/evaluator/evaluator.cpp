#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/kendalls_tau.hpp"
#include <cstddef>
#include <vector>

namespace evaluator {

Scoretable evaluate_simplification(void /*Original, simplified, datastructure tbd*/) {

    Scoretable eval_score;
    std::vector<std::size_t> original_knn_idxs;
    std::vector<std::size_t> simplified_knn_idxs;
    eval_score.knn_score = knn::evaluate_knn(original_knn_idxs, simplified_knn_idxs);

    return eval_score;
}

} // namespace evaluator
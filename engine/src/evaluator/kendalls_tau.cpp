#include "backend/engine/evaluator/kendalls_tau.hpp"
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>
namespace evaluator::knn {
namespace {
std::vector<std::pair<size_t, size_t>> shared_ranks(const std::vector<size_t> &original,
                                                    const std::vector<size_t> &simplified) {

    std::vector<std::pair<size_t, size_t>> shared;

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
} // namespace

double evaluate_knn(const std::vector<size_t> &original, const std::vector<size_t> &simplified) {
    std::vector<std::pair<size_t, size_t>> overlap = shared_ranks(original, simplified);
    if (overlap.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::size_t concordant = 0, discordant = 0;

    for (std::size_t i = 0; i < overlap.size(); i++) {
        for (std::size_t j = i + 1; j < overlap.size(); j++) {
            if (overlap[i].second < overlap[j].second) {
                concordant++;
            } else {
                discordant++;
            }
        }
    }
    double tau = (double(concordant) - double(discordant)) / double(concordant + discordant);
    return tau;
}

} // namespace evaluator::knn
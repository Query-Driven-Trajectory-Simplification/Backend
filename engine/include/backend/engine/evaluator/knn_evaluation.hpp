#pragma once

#include <cstddef>
#include <vector>
namespace evaluator::knn {

double kendalls_tau(const std::vector<std::size_t> &original,
                    const std::vector<std::size_t> &simplified);

// Share of original that also appears in simplified, ignoring order. NaN if original is empty.
double recall_at_k(const std::vector<std::size_t> &original,
                   const std::vector<std::size_t> &simplified);

} // namespace evaluator::knn

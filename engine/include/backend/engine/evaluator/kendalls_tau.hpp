#pragma once

#include <cstddef>
#include <vector>
namespace evaluator::knn {

double kendalls_tau(const std::vector<std::size_t> &original,
                    const std::vector<std::size_t> &simplified);

}
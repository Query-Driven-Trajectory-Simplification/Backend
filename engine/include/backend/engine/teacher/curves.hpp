#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace teacher {

// What one trajectory is worth at each budget: value[k] and method[k] are for b = 2 + k points,
// endpoints included. Fréchet thresholds flip on and off, so a curve is a step function and need
// not be concave or even increasing
struct Curve {
    std::vector<double> value;
    std::vector<std::uint8_t> method; // the candidate that reached value[k]

    std::size_t max_budget() const { return value.size() + 1; }
    double at(std::size_t b) const { return value[b - 2]; }
};

// Maximises the sum of curves[i].at(b_i) subject to sum b_i = total_budget, 2 <= b_i <=
// max_budget, greedily from b_i = 2 on the best gain per point over the next 1 to 3 points, which
// steps over flat stretches plain marginal greedy would stall on. Writes the curve sum to value.
// Throws if the curves cannot hold the budget
std::vector<std::size_t> allocate(const std::vector<Curve> &curves, std::size_t total_budget,
                                  double *value = nullptr);

} // namespace teacher

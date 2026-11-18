#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace teacher {

// One trajectory's value at b = 2 + k points. A step function, not necessarily concave or
// increasing
struct Curve {
    std::vector<double> value;
    std::vector<std::uint8_t> method; // the candidate that reached value[k]

    std::size_t max_budget() const { return value.size() + 1; }
    double at(std::size_t budget) const { return value[budget - 2]; }
};

// Maximises sum curves[i].at(b_i) with sum b_i = total_budget, greedily on the best gain per point
// over the next 1 to 3 points, which steps over flat stretches. Throws if the budget cannot fit
std::vector<std::size_t> allocate(const std::vector<Curve> &curves, std::size_t total_budget,
                                  double *value = nullptr);

} // namespace teacher

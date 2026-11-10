#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace teacher {

// What one trajectory is worth at each budget: value[k] and method[k] are for b = 2 + k points,
// endpoints included. Fréchet thresholds flip on and off, so a curve is a step function and need
// not be concave or even increasing. Selection reads it at a fixed budget; an allocation search
// over the curves (how many points each trajectory gets) can be added on top later
struct Curve {
    std::vector<double> value;
    std::vector<std::uint8_t> method; // the candidate that reached value[k]

    std::size_t max_budget() const { return value.size() + 1; }
    double at(std::size_t b) const { return value[b - 2]; }
};

} // namespace teacher

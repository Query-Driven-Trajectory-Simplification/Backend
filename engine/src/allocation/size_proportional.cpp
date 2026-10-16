#include "backend/engine/allocation/size_proportional.hpp"
#include "backend/engine/allocation/budget_allocator.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <queue>
#include <stdexcept>
#include <vector>
#include <format>

namespace allocation {
namespace {

struct TripRatio {
    std::size_t trip_index;
    std::size_t budget;
    std::size_t size;

    // a/b < c/d  <=>  a*d < c*b
    // no need for float on size_t
    bool operator<(const TripRatio &other) const {
        return budget * other.size < other.budget * size;
    }
    bool operator>(const TripRatio &other) const {
        return budget * other.size > other.budget * size;
    }
};

void DistributeLeftovers(const std::vector<std::size_t> &sizes, std::size_t total_budget,
                         std::size_t &allocated, std::vector<std::size_t> &budgets) {
    std::priority_queue<TripRatio, std::vector<TripRatio>, std::greater<TripRatio>> min_ratio;

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (budgets[i] < sizes[i]){
            min_ratio.push({i, budgets[i], sizes[i]});
        }
    }

    while (allocated < total_budget && !min_ratio.empty()) {
        TripRatio top_element = min_ratio.top();
        min_ratio.pop();
        budgets[top_element.trip_index]++;
        top_element.budget++;
        allocated++;
        if (top_element.budget < top_element.size) {
            min_ratio.push(top_element);
        }
    }
}

void TakeBackFromOvershoot(const std::vector<std::size_t> &sizes, std::size_t total_budget,
                           std::size_t &allocated, std::vector<std::size_t> &budgets) {

    std::priority_queue<TripRatio, std::vector<TripRatio>, std::less<TripRatio>> max_ratio;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (budgets[i] > kMinPointsPerTrip){
            max_ratio.push({i, budgets[i], sizes[i]});
        }
    }

    while (allocated > total_budget && !max_ratio.empty()) {
        TripRatio top_element = max_ratio.top();
        max_ratio.pop();
        budgets[top_element.trip_index]--;
        top_element.budget--;
        allocated--;
        if (top_element.budget > kMinPointsPerTrip) {
            max_ratio.push(top_element);
        }
    }
}

} // namespace

std::vector<std::size_t> proportional_budget(const core::TrajectoryDb &trajectories,
                                             std::size_t total_budget) {
    const std::size_t num_trips = trajectories.size();
    const std::size_t num_points = trajectories.num_points();

    if (total_budget < kMinPointsPerTrip * num_trips) {
        throw std::invalid_argument("Total budget must be at least 2 points per trip");
    }

    std::vector<std::size_t> sizes(num_trips);
    for (std::size_t i = 0; i < num_trips; ++i) {
        sizes[i] = trajectories.trajectory(i).size();
        if (sizes[i] < kMinPointsPerTrip){
            throw std::invalid_argument(std::format("Trip {} No trip may be less than 2 points", i));
        }
    }

    if (total_budget >= num_points) {
        return sizes;
    }

    std::vector<std::size_t> budgets(num_trips);
    std::size_t allocated = 0;
    for (std::size_t i = 0; i < num_trips; ++i) {
        const std::size_t share = sizes[i] * total_budget / num_points;
        budgets[i] = std::clamp(share, kMinPointsPerTrip, sizes[i]);
        allocated += budgets[i];
    }

    if (allocated > total_budget) {
        TakeBackFromOvershoot(sizes, total_budget, allocated, budgets);
    } else if (allocated < total_budget) {
        DistributeLeftovers(sizes, total_budget, allocated, budgets);
    }
    return budgets;
}

} // namespace allocation

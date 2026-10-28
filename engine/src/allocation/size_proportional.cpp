#include "backend/engine/allocation/size_proportional.hpp"
#include "backend/engine/allocation/constants.hpp"
#include <algorithm>
#include <cstddef>
#include <format>
#include <functional>
#include <queue>
#include <stdexcept>
#include <vector>

namespace allocation {
namespace {

struct TrajectoryRatio {
    std::size_t trajectory_index;
    std::size_t budget;
    std::size_t size;

    // a/b < c/d  <=>  a*d < c*b
    // no need for float on size_t
    bool operator<(const TrajectoryRatio &other) const {
        return budget * other.size < other.budget * size;
    }
    bool operator>(const TrajectoryRatio &other) const { return other < *this; }
};

void distribute_leftovers(const std::vector<std::size_t> &sizes, std::size_t total_budget,
                          std::size_t &allocated, std::vector<std::size_t> &budgets) {
    std::priority_queue<TrajectoryRatio, std::vector<TrajectoryRatio>,
                        std::greater<TrajectoryRatio>>
        min_ratio;

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (budgets[i] < sizes[i]) {
            min_ratio.push({i, budgets[i], sizes[i]});
        }
    }

    while (allocated < total_budget && !min_ratio.empty()) {
        TrajectoryRatio top_element = min_ratio.top();
        min_ratio.pop();
        budgets[top_element.trajectory_index]++;
        top_element.budget++;
        allocated++;
        if (top_element.budget < top_element.size) {
            min_ratio.push(top_element);
        }
    }
}

void take_back_from_overshoot(const std::vector<std::size_t> &sizes, std::size_t total_budget,
                              std::size_t &allocated, std::vector<std::size_t> &budgets) {

    std::priority_queue<TrajectoryRatio, std::vector<TrajectoryRatio>, std::less<TrajectoryRatio>>
        max_ratio;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (budgets[i] > kMinPointsPerTrajectory) {
            max_ratio.push({i, budgets[i], sizes[i]});
        }
    }

    while (allocated > total_budget && !max_ratio.empty()) {
        TrajectoryRatio top_element = max_ratio.top();
        max_ratio.pop();
        budgets[top_element.trajectory_index]--;
        top_element.budget--;
        allocated--;
        if (top_element.budget > kMinPointsPerTrajectory) {
            max_ratio.push(top_element);
        }
    }
}

} // namespace

std::vector<std::size_t> proportional_budget(const core::TrajectoryDb &trajectories,
                                             std::size_t total_budget) {
    const std::size_t num_trajectories = trajectories.size();
    const std::size_t num_points = trajectories.num_points();

    if (total_budget < kMinPointsPerTrajectory * num_trajectories) {
        throw std::invalid_argument(std::format(
            "proportional_budget: total budget must be at least {} points per trajectory",
            kMinPointsPerTrajectory));
    }

    std::vector<std::size_t> sizes(num_trajectories);
    for (std::size_t i = 0; i < num_trajectories; ++i) {
        sizes[i] = trajectories.trajectory(i).size();
        if (sizes[i] < kMinPointsPerTrajectory) {
            throw std::invalid_argument(std::format(
                "proportional_budget: trajectory {} has {} points, at least {} required",
                trajectories.trajectory(i).trajectory_id, sizes[i], kMinPointsPerTrajectory));
        }
    }

    if (total_budget >= num_points) {
        return sizes;
    }

    std::vector<std::size_t> budgets(num_trajectories);
    std::size_t allocated = 0;
    for (std::size_t i = 0; i < num_trajectories; ++i) {
        const std::size_t share = sizes[i] * total_budget / num_points;
        budgets[i] = std::clamp(share, kMinPointsPerTrajectory, sizes[i]);
        allocated += budgets[i];
    }

    if (allocated > total_budget) {
        take_back_from_overshoot(sizes, total_budget, allocated, budgets);
    } else if (allocated < total_budget) {
        distribute_leftovers(sizes, total_budget, allocated, budgets);
    }
    return budgets;
}

} // namespace allocation

#include "backend/engine/allocation/size_proportional.hpp"
#include "backend/engine/allocation/constants.hpp"
#include <algorithm>
#include <cmath>
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
    double weight;

    // a/b < c/d  <=>  a*d < c*b
    // Exact while the products stay below 2^53, so alpha = 1 gives the old integer results.
    bool operator<(const TrajectoryRatio &other) const {
        return static_cast<double>(budget) * other.weight <
               static_cast<double>(other.budget) * weight;
    }
    bool operator>(const TrajectoryRatio &other) const { return other < *this; }
};

void distribute_leftovers(const std::vector<std::size_t> &sizes, std::size_t total_budget,
                          std::size_t &allocated, std::vector<std::size_t> &budgets,
                          const std::vector<double> &weights) {
    std::priority_queue<TrajectoryRatio, std::vector<TrajectoryRatio>,
                        std::greater<TrajectoryRatio>>
        min_ratio;

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (budgets[i] < sizes[i]) {
            min_ratio.push({i, budgets[i], weights[i]});
        }
    }

    while (allocated < total_budget && !min_ratio.empty()) {
        TrajectoryRatio top_element = min_ratio.top();
        min_ratio.pop();
        budgets[top_element.trajectory_index]++;
        top_element.budget++;
        allocated++;
        if (top_element.budget < sizes[top_element.trajectory_index]) {
            min_ratio.push(top_element);
        }
    }
}

void take_back_from_overshoot(const std::vector<std::size_t> &sizes, std::size_t total_budget,
                              std::size_t &allocated, std::vector<std::size_t> &budgets,
                              const std::vector<double> &weights) {

    std::priority_queue<TrajectoryRatio, std::vector<TrajectoryRatio>, std::less<TrajectoryRatio>>
        max_ratio;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (budgets[i] > kMinPointsPerTrajectory) {
            max_ratio.push({i, budgets[i], weights[i]});
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
                                             std::size_t total_budget, double alpha) {
    if (!std::isfinite(alpha))
        throw std::invalid_argument("proportional_budget: alpha must be finite");
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

    std::vector<double> weights(num_trajectories);
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < num_trajectories; ++i) {
        weights[i] = std::pow(static_cast<double>(sizes[i]), alpha);
        weight_sum += weights[i];
    }
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0)
        throw std::invalid_argument("proportional_budget: alpha out of range for these sizes");

    std::vector<std::size_t> budgets(num_trajectories);
    std::size_t allocated = 0;
    for (std::size_t i = 0; i < num_trajectories; ++i) {
        // Multiply before dividing: with alpha 1: every value if an exact integer, so the floor
        // equals the old sizes[i] * total_budget / num_points
        const auto share =
            static_cast<std::size_t>(static_cast<double>(total_budget) * weights[i] / weight_sum);
        budgets[i] = std::clamp(share, kMinPointsPerTrajectory, sizes[i]);
        allocated += budgets[i];
    }

    if (allocated > total_budget) {
        take_back_from_overshoot(sizes, total_budget, allocated, budgets, weights);
    } else if (allocated < total_budget) {
        distribute_leftovers(sizes, total_budget, allocated, budgets, weights);
    }
    return budgets;
}
} // namespace allocation

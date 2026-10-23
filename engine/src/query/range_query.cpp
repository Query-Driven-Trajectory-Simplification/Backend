#include "backend/engine/query/range_query.hpp"
#include "backend/engine/core/trajectory_db.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace query {

namespace {

void validate_config(const RangeConfig &config) {
    if (!std::isfinite(config.box_size_m) || config.box_size_m <= 0)
        throw std::invalid_argument("range: box_size_m must be finite and positive");
    if (config.window_seconds < 0)
        throw std::invalid_argument("range: window_seconds must be nonnegative");
    if (config.min_results == 0)
        throw std::invalid_argument("range: min_results must be positive");
    if (config.attempts_per_query == 0)
        throw std::invalid_argument("range: attempts_per_query must be positive");
    if (config.query_count > std::numeric_limits<std::size_t>::max() / config.attempts_per_query)
        throw std::invalid_argument("range: query_count * attempts_per_query overflows");
}

bool in_box(const RangeBox &box, std::int64_t t, double x, double y) {
    return t >= box.t_min && t <= box.t_max && x >= box.x_min && x <= box.x_max && y >= box.y_min &&
           y <= box.y_max;
}

// Number of points before each trajectory, so a point index can be mapped to its trajectory
std::vector<std::size_t> point_offsets(const core::TrajectoryDb &trajectories) {
    std::vector<std::size_t> offsets{0};
    offsets.reserve(trajectories.size() + 1);
    for (std::size_t i = 0; i < trajectories.size(); i++)
        offsets.push_back(offsets.back() + trajectories.trajectory(i).size());
    return offsets;
}

RangeBox box_around(const core::TrajectoryView &trajectory, std::size_t point,
                    const RangeConfig &config) {
    const double half_size = config.box_size_m / 2;
    const std::int64_t before = config.window_seconds / 2;
    const std::int64_t after = config.window_seconds - before;
    return {
        .x_min = trajectory.x[point] - half_size,
        .x_max = trajectory.x[point] + half_size,
        .y_min = trajectory.y[point] - half_size,
        .y_max = trajectory.y[point] + half_size,
        .t_min = trajectory.t[point] - before,
        .t_max = trajectory.t[point] + after,
    };
}

bool accept(std::size_t results, const RangeConfig &config, std::mt19937_64 &rng) {
    if (results < config.min_results)
        return false;
    if (results >= config.full_accept_at)
        return true;
    const double random = std::generate_canonical<double, 53>(rng);
    return random * static_cast<double>(config.full_accept_at) < static_cast<double>(results);
}

} // namespace

std::vector<std::size_t> range(const core::TrajectoryDb &trajectories, const RangeBox &box) {
    std::vector<std::size_t> matches;
    for (std::size_t i = 0; i < trajectories.size(); i++) {
        const auto trajectory = trajectories.trajectory(i);
        const auto first = std::ranges::lower_bound(trajectory.t, box.t_min) - trajectory.t.begin();
        for (auto j = static_cast<std::size_t>(first);
             j < trajectory.size() && trajectory.t[j] <= box.t_max; j++) {
            if (in_box(box, trajectory.t[j], trajectory.x[j], trajectory.y[j])) {
                matches.push_back(i);
                break;
            }
        }
    }
    return matches;
}

RangeWorkload generate_range_queries(const core::TrajectoryDb &original,
                                     const RangeConfig &config) {
    validate_config(config);

    RangeWorkload result;
    const auto offsets = point_offsets(original);
    const std::size_t num_points = offsets.back();
    if (num_points == 0)
        return result;

    std::mt19937_64 rng(config.seed);
    std::uniform_int_distribution<std::size_t> random_point(0, num_points - 1);
    std::unordered_set<std::size_t> used_centres;
    const std::size_t max_attempts = config.query_count * config.attempts_per_query;
    for (std::size_t attempt = 0;
         attempt < max_attempts && result.queries.size() < config.query_count &&
         used_centres.size() < num_points;
         attempt++) {
        const std::size_t point = random_point(rng);
        if (used_centres.contains(point))
            continue;

        const auto trajectory_index = static_cast<std::size_t>(
            std::ranges::upper_bound(offsets, point) - offsets.begin() - 1);
        const auto box = box_around(original.trajectory(trajectory_index),
                                    point - offsets[trajectory_index], config);
        auto reference = range(original, box);
        // Too few results can never improve for this centre; random rejections can be retried.
        if (reference.size() < config.min_results) {
            used_centres.insert(point);
            continue;
        }
        if (!accept(reference.size(), config, rng))
            continue;

        used_centres.insert(point);
        result.queries.push_back({box, std::move(reference)});
    }
    return result;
}

} // namespace query

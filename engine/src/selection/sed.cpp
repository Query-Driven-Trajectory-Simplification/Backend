#include "backend/engine/selection/sed.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <span>
#include <vector>

namespace selection::sed {

namespace {

struct Point {
    std::size_t prev;
    std::size_t next;
};

struct Removal {
    double cost;
    std::size_t index;
    std::size_t version;

    bool operator<(const Removal &other) const { return cost > other.cost; }
};

double synchronous_factor(double point_time, double start_time, double end_time) {
    return (point_time - start_time) / (end_time - start_time);
}

double squared_synchronous_distance(const core::TrajectoryView &trajectory, std::size_t first,
                                    std::size_t last, std::size_t point) {
    const std::span<const std::int64_t> ts = trajectory.t;
    const std::span<const double> xs = trajectory.x;
    const std::span<const double> ys = trajectory.y;

    const double c =
        synchronous_factor(static_cast<double>(ts[point]), static_cast<double>(ts[first]),
                           static_cast<double>(ts[last]));

    const double dx = xs[first] + c * (xs[last] - xs[first]) - xs[point];
    const double dy = ys[first] + c * (ys[last] - ys[first]) - ys[point];
    return dx * dx + dy * dy;
}

double span_error(const core::TrajectoryView &trajectory, std::size_t first, std::size_t last) {
    const std::span<const std::int64_t> ts = trajectory.t;
    const std::span<const double> xs = trajectory.x;
    const std::span<const double> ys = trajectory.y;

    const double t0 = static_cast<double>(ts[first]);
    const double inv_dt = 1.0 / (static_cast<double>(ts[last]) - t0);
    const double x0 = xs[first];
    const double y0 = ys[first];
    const double seg_dx = xs[last] - x0;
    const double seg_dy = ys[last] - y0;

    double error = 0.0;
    for (std::size_t k = first + 1; k < last; ++k) {
        const double c = (static_cast<double>(ts[k]) - t0) * inv_dt;
        const double dx = x0 + c * seg_dx - xs[k];
        const double dy = y0 + c * seg_dy - ys[k];
        error = std::max(error, dx * dx + dy * dy);
    }
    return error;
}

} // namespace

double synchronous_distance(const core::TrajectoryView &trajectory, std::size_t first,
                            std::size_t last, std::size_t point) {
    return std::sqrt(squared_synchronous_distance(trajectory, first, last, point));
}

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep) {
    const std::size_t n = trajectory.size();

    if (points_to_keep >= n || n <= 2) {
        std::vector<std::size_t> all_points_kept(n);
        for (std::size_t i = 0; i < n; ++i)
            all_points_kept[i] = i;
        return all_points_kept;
    }

    std::vector<Point> points(n);
    for (std::size_t i = 0; i < n; ++i)
        points[i] = {.prev = i - 1, .next = i + 1};

    std::vector<bool> kept_indices(n, true);
    std::vector<std::size_t> versions(n, 0);
    std::size_t kept_points = n;

    std::priority_queue<Removal> heap;
    for (std::size_t i = 1; i < n - 1; ++i)
        heap.push({.cost = span_error(trajectory, i - 1, i + 1), .index = i, .version = 0});

    while (kept_points > points_to_keep && !heap.empty()) {
        const Removal top = heap.top();
        heap.pop();
        if (!kept_indices[top.index] || top.version != versions[top.index])
            continue;

        const std::size_t prev = points[top.index].prev;
        const std::size_t next = points[top.index].next;
        kept_indices[top.index] = false;
        points[prev].next = next;
        points[next].prev = prev;
        kept_points--;

        for (const std::size_t curr : {prev, next}) {
            if (curr == 0 || curr == n - 1)
                continue;
            versions[curr]++;
            heap.push({.cost = span_error(trajectory, points[curr].prev, points[curr].next),
                       .index = curr,
                       .version = versions[curr]});
        }
    }

    std::vector<std::size_t> simplified_trajectory;
    simplified_trajectory.reserve(kept_points);
    for (std::size_t i = 0; i < n; ++i) {
        if (kept_indices[i])
            simplified_trajectory.push_back(i);
    }
    return simplified_trajectory;
}

} // namespace selection::sed

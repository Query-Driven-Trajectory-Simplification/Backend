#include "backend/engine/selection/rdp.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <span>

namespace selection::rdp {

namespace {

// https://www.youtube.com/watch?v=SbVXh5VtxKw
// Great vid for Ramer-Douglas-Peucker
// The implementation uses max-prio queue with fixed point budget instead of standard epsilon error
// like in vid.

double segment_distance(double px, double py, double ax, double ay, double bx, double by) {
    const double dx = bx - ax;
    const double dy = by - ay;
    const double len_sq = dx * dx + dy * dy;
    double s = 0.0;
    if (len_sq > 0.0)
        s = std::clamp(((px - ax) * dx + (py - ay) * dy) / len_sq, 0.0, 1.0);
    return std::hypot(px - (ax + s * dx), py - (ay + s * dy));
}

struct Split {
    double error;
    std::size_t index;
    std::size_t first;
    std::size_t last;

    bool operator<(const Split &other) const { return error < other.error; }
};

void push_split(std::priority_queue<Split> &heap, std::span<const double> x,
                std::span<const double> y, std::size_t first, std::size_t last) {
    if (last - first < 2)
        return;
    Split split{-1.0, first + 1, first, last};
    for (std::size_t k = first + 1; k < last; ++k) {
        const double d = segment_distance(x[k], y[k], x[first], y[first], x[last], y[last]);
        if (d > split.error) {
            split.error = d;
            split.index = k;
        }
    }
    heap.push(split);
}

} // namespace

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep) {
    const std::size_t n = trajectory.size();
    const auto &x = trajectory.x;
    const auto &y = trajectory.y;

    if (points_to_keep >= n) {
        std::vector<std::size_t> all_points_kept(n);
        for (std::size_t i = 0; i < n; ++i)
            all_points_kept[i] = i;
        return all_points_kept;
    }
    std::vector<bool> kept(n, false);
    kept.front() = kept.back() = true;
    std::size_t num_kept = 2;

    std::priority_queue<Split> heap;

    push_split(heap, x, y, 0, n - 1);
    while (num_kept < points_to_keep && !heap.empty()) {
        const Split split = heap.top();
        heap.pop();
        kept[split.index] = true;
        ++num_kept;
        push_split(heap, x, y, split.first, split.index);
        push_split(heap, x, y, split.index, split.last);
    }

    std::vector<std::size_t> indices;
    indices.reserve(num_kept);
    for (std::size_t i = 0; i < n; ++i) {
        if (kept[i])
            indices.push_back(i);
    }
    return indices;
}

} // namespace selection::rdp
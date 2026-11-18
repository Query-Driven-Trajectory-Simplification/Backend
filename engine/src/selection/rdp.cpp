#include "backend/engine/selection/rdp.hpp"

#include <algorithm>
#include <queue>
#include <span>

namespace selection::rdp {

namespace {

// https://www.youtube.com/watch?v=SbVXh5VtxKw
// Great vid for Ramer-Douglas-Peucker
// The implementation uses max-prio queue with fixed point budget instead of standard epsilon error
// like in vid.

double squared_segment_distance(double px, double py, double ax, double ay, double bx, double by) {
    const double dx = bx - ax;
    const double dy = by - ay;
    const double len_sq = dx * dx + dy * dy;
    double s = 0.0;
    if (len_sq > 0.0)
        s = std::clamp(((px - ax) * dx + (py - ay) * dy) / len_sq, 0.0, 1.0);
    const double ex = px - (ax + s * dx);
    const double ey = py - (ay + s * dy);
    return ex * ex + ey * ey;
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
        const double d = squared_segment_distance(x[k], y[k], x[first], y[first], x[last], y[last]);
        if (d > split.error) {
            split.error = d;
            split.index = k;
        }
    }
    heap.push(split);
}

} // namespace

std::vector<std::size_t> sequence(const core::TrajectoryView &trajectory, std::size_t limit) {
    const std::size_t n = trajectory.size();
    if (n == 0)
        return {};
    if (n == 1)
        return {0};
    limit = std::clamp(limit, std::size_t{2}, n);
    std::vector<std::size_t> indices{0, n - 1};
    indices.reserve(limit);
    std::priority_queue<Split> heap;
    push_split(heap, trajectory.x, trajectory.y, 0, n - 1);
    while (indices.size() < limit && !heap.empty()) {
        const Split split = heap.top();
        heap.pop();
        indices.push_back(split.index);
        push_split(heap, trajectory.x, trajectory.y, split.first, split.index);
        push_split(heap, trajectory.x, trajectory.y, split.index, split.last);
    }
    return indices;
}

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep) {
    if (points_to_keep >= trajectory.size()) {
        std::vector<std::size_t> indices(trajectory.size());
        for (std::size_t i = 0; i < indices.size(); ++i)
            indices[i] = i;
        return indices;
    }
    auto indices = sequence(trajectory, points_to_keep);
    std::ranges::sort(indices);
    return indices;
}

} // namespace selection::rdp

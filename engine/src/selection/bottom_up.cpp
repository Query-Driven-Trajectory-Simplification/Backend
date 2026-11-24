#include "backend/engine/selection/bottom_up.hpp"
#include "backend/engine/selection/distance_functions.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <vector>

namespace selection::bottom_up {

namespace {

struct Neighbours {
    std::size_t prev;
    std::size_t next;
};

struct Removal {
    double cost;
    std::size_t index;
    std::size_t version;

    bool operator<(const Removal &other) const { return cost > other.cost; }
};

double span_cost(const core::TrajectoryView &trajectory, const std::vector<double> &headings,
                 std::size_t first, std::size_t last, Measure measure) {
    switch (measure) {
    case Measure::Sed:
        return distance::max_sed_squared(trajectory, first, last);
    case Measure::Ped:
        return distance::max_ped_squared(trajectory, first, last);
    case Measure::Dad:
        return distance::max_heading_change(trajectory, headings, first, last);
    }
    throw std::invalid_argument("bottom-up: unknown measure");
}

std::vector<double> headings_for(const core::TrajectoryView &trajectory, Measure measure) {
    return measure == Measure::Dad ? distance::segment_headings(trajectory) : std::vector<double>{};
}

std::vector<std::size_t> simplify_impl(const core::TrajectoryView &trajectory,
                                       std::size_t points_to_keep,
                                       std::vector<std::size_t> *removed, Measure measure) {
    const std::size_t n = trajectory.size();

    if (points_to_keep >= n || n <= 2) {
        std::vector<std::size_t> all_points_kept(n);
        for (std::size_t i = 0; i < n; ++i)
            all_points_kept[i] = i;
        return all_points_kept;
    }

    const auto headings = headings_for(trajectory, measure);
    auto cost = [&](std::size_t first, std::size_t last) {
        return span_cost(trajectory, headings, first, last, measure);
    };

    std::vector<Neighbours> points(n);
    for (std::size_t i = 0; i < n; ++i)
        points[i] = {.prev = i - 1, .next = i + 1};

    std::vector<bool> kept_indices(n, true);
    std::vector<std::size_t> versions(n, 0);
    std::size_t kept_points = n;

    std::priority_queue<Removal> heap;
    for (std::size_t i = 1; i < n - 1; ++i)
        heap.push({.cost = cost(i - 1, i + 1), .index = i, .version = 0});

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
        if (removed)
            removed->push_back(top.index);

        for (const std::size_t curr : {prev, next}) {
            if (curr == 0 || curr == n - 1)
                continue;
            versions[curr]++;
            heap.push({.cost = cost(points[curr].prev, points[curr].next),
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

} // namespace

double span_error(const core::TrajectoryView &trajectory, std::size_t first, std::size_t last,
                  Measure measure) {
    if (first >= last || last >= trajectory.size())
        throw std::invalid_argument("bottom-up: span needs first < last < size");
    const double cost =
        span_cost(trajectory, headings_for(trajectory, measure), first, last, measure);
    return measure == Measure::Dad ? cost : std::sqrt(cost);
}

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep, Measure measure) {
    return simplify_impl(trajectory, points_to_keep, nullptr, measure);
}

std::vector<std::size_t> sequence(const core::TrajectoryView &trajectory, std::size_t limit,
                                  Measure measure) {
    const auto n = trajectory.size();
    if (n <= 2)
        return simplify(trajectory, n, measure);
    limit = std::clamp(limit, std::size_t{2}, n);
    std::vector<std::size_t> removed;
    removed.reserve(n - 2);
    auto indices = simplify_impl(trajectory, 2, &removed, measure);
    indices.reserve(limit);
    // Reverse removal order: the last removed point is the first to restore above budget two.
    indices.insert(indices.end(), removed.rbegin(), removed.rbegin() + (limit - 2));
    return indices;
}

} // namespace selection::bottom_up

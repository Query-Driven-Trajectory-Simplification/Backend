#include "backend/engine/selection/bottom_up.hpp"
#include "backend/engine/core/parallel.hpp"
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
                                       std::vector<std::size_t> *removed,
                                       std::vector<double> *removed_costs, Measure measure) {
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
        if (removed_costs)
            removed_costs->push_back(top.cost);

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
    return simplify_impl(trajectory, points_to_keep, nullptr, nullptr, measure);
}

std::vector<std::size_t> sequence(const core::TrajectoryView &trajectory, std::size_t limit,
                                  Measure measure) {
    const auto n = trajectory.size();
    if (n <= 2)
        return simplify(trajectory, n, measure);
    limit = std::clamp(limit, std::size_t{2}, n);
    std::vector<std::size_t> removed;
    removed.reserve(n - 2);
    auto indices = simplify_impl(trajectory, 2, &removed, nullptr, measure);
    indices.reserve(limit);
    // Reverse removal order: the last removed point is the first to restore above budget two.
    indices.insert(indices.end(), removed.rbegin(), removed.rbegin() + (limit - 2));
    return indices;
}

core::Simplification simplify_database(const core::TrajectoryDb &db, std::size_t total_budget,
                                       Measure measure) {
    const std::size_t n = db.size();
    std::size_t endpoints = 0;
    for (std::size_t i = 0; i < n; ++i)
        endpoints += std::min<std::size_t>(db.trajectory(i).size(), 2);
    if (total_budget < endpoints || total_budget > db.num_points())
        throw std::invalid_argument(
            "global bottom-up: budget must be between endpoints and all points");

    std::vector<std::vector<std::size_t>> removed(n);
    std::vector<std::vector<double>> costs(n);
    core::parallel_for(n, [&](std::size_t i) {
        simplify_impl(db.trajectory(i), 2, &removed[i], &costs[i], measure);
    });

    struct Next {
        double cost;
        std::size_t trajectory;
        std::size_t position;

        bool operator<(const Next &other) const {
            if (cost != other.cost)
                return cost > other.cost;
            return trajectory > other.trajectory;
        }
    };
    std::priority_queue<Next> heap;
    for (std::size_t i = 0; i < n; ++i)
        if (!costs[i].empty())
            heap.push({costs[i][0], i, 0});

    std::vector<std::size_t> taken(n, 0);
    std::size_t kept = db.num_points();
    while (kept > total_budget) {
        const Next top = heap.top();
        heap.pop();
        ++taken[top.trajectory];
        --kept;
        if (top.position + 1 < costs[top.trajectory].size())
            heap.push({costs[top.trajectory][top.position + 1], top.trajectory, top.position + 1});
    }

    core::Simplification simplification;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<char> keep(db.trajectory(i).size(), 1);
        for (std::size_t k = 0; k < taken[i]; ++k)
            keep[removed[i][k]] = 0;
        std::vector<std::size_t> indices;
        for (std::size_t k = 0; k < keep.size(); ++k)
            if (keep[k])
                indices.push_back(k);
        simplification.add_trajectory(indices);
    }
    return simplification;
}

} // namespace selection::bottom_up

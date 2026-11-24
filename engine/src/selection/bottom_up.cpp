#include "backend/engine/selection/bottom_up.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <queue>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace selection::bottom_up {

namespace {

// Each cost returns a value that orders spans like their error (squared distances for Sed and
// Ped), and error(cost) turns it back into the error

// Max squared SED of the points strictly between first and last
struct SedCost {
    const core::TrajectoryView &trajectory;

    double operator()(std::size_t first, std::size_t last) const {
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
    static double error(double cost) { return std::sqrt(cost); }
};

// Max squared distance of the points strictly between first and last to the infinite line
// through them; 0 when first and last coincide, as in MLSimp's ped_op
struct PedCost {
    const core::TrajectoryView &trajectory;

    double operator()(std::size_t first, std::size_t last) const {
        const std::span<const double> xs = trajectory.x;
        const std::span<const double> ys = trajectory.y;
        const double x0 = xs[first];
        const double y0 = ys[first];
        const double seg_dx = xs[last] - x0;
        const double seg_dy = ys[last] - y0;
        const double length_sq = seg_dx * seg_dx + seg_dy * seg_dy;
        if (length_sq == 0.0)
            return 0.0;
        double cross_sq = 0.0;
        for (std::size_t k = first + 1; k < last; ++k) {
            const double cross = seg_dx * (ys[k] - y0) - seg_dy * (xs[k] - x0);
            cross_sq = std::max(cross_sq, cross * cross);
        }
        return cross_sq / length_sq;
    }
    static double error(double cost) { return std::sqrt(cost); }
};

// Heading of a -> b in [0, 2 pi); atan2(0, 0) = 0 for repeated points, as in MLSimp's angle
double heading(double dx, double dy) {
    const double angle = std::atan2(dy, dx);
    return angle >= 0.0 ? angle : 2.0 * std::numbers::pi + angle;
}

// Max circular angle between the heading of first -> last and of each original segment
// k -> k + 1 from first to last, as in MLSimp's dad_op. Headings are computed once
struct DadCost {
    const core::TrajectoryView &trajectory;
    std::vector<double> headings; // headings[k]: original segment k -> k + 1

    explicit DadCost(const core::TrajectoryView &t) : trajectory(t) {
        const std::size_t n = t.size();
        headings.resize(n > 0 ? n - 1 : 0);
        for (std::size_t k = 0; k + 1 < n; ++k)
            headings[k] = heading(t.x[k + 1] - t.x[k], t.y[k + 1] - t.y[k]);
    }

    double operator()(std::size_t first, std::size_t last) const {
        const double theta = heading(trajectory.x[last] - trajectory.x[first],
                                     trajectory.y[last] - trajectory.y[first]);
        double error = 0.0;
        for (std::size_t k = first; k < last; ++k) {
            const double gap = std::abs(theta - headings[k]);
            error = std::max(error, std::min(gap, 2.0 * std::numbers::pi - gap));
        }
        return error;
    }
    static double error(double cost) { return cost; }
};

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

template <class Cost>
std::vector<std::size_t> simplify_impl(const core::TrajectoryView &trajectory,
                                       std::size_t points_to_keep,
                                       std::vector<std::size_t> *removed, const Cost &cost) {
    const std::size_t n = trajectory.size();

    if (points_to_keep >= n || n <= 2) {
        std::vector<std::size_t> all_points_kept(n);
        for (std::size_t i = 0; i < n; ++i)
            all_points_kept[i] = i;
        return all_points_kept;
    }

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

template <class Cost>
std::vector<std::size_t> sequence_impl(const core::TrajectoryView &trajectory, std::size_t limit,
                                       const Cost &cost) {
    const auto n = trajectory.size();
    if (n <= 2)
        return simplify_impl(trajectory, n, nullptr, cost);
    limit = std::clamp(limit, std::size_t{2}, n);
    std::vector<std::size_t> removed;
    removed.reserve(n - 2);
    auto indices = simplify_impl(trajectory, 2, &removed, cost);
    indices.reserve(limit);
    // Reverse removal order: the last removed point is the first to restore above budget two.
    indices.insert(indices.end(), removed.rbegin(), removed.rbegin() + (limit - 2));
    return indices;
}

// Calls f with the cost functor for measure
template <class F>
decltype(auto) with_cost(const core::TrajectoryView &trajectory, Measure measure, F &&f) {
    switch (measure) {
    case Measure::Sed:
        return f(SedCost{trajectory});
    case Measure::Ped:
        return f(PedCost{trajectory});
    case Measure::Dad:
        return f(DadCost{trajectory});
    }
    throw std::invalid_argument("bottom-up: unknown measure");
}

} // namespace

double span_error(const core::TrajectoryView &trajectory, std::size_t first, std::size_t last,
                  Measure measure) {
    if (first >= last || last >= trajectory.size())
        throw std::invalid_argument("bottom-up: span needs first < last < size");
    return with_cost(trajectory, measure, [&](const auto &cost) {
        return std::remove_cvref_t<decltype(cost)>::error(cost(first, last));
    });
}

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep, Measure measure) {
    return with_cost(trajectory, measure, [&](const auto &cost) {
        return simplify_impl(trajectory, points_to_keep, nullptr, cost);
    });
}

std::vector<std::size_t> sequence(const core::TrajectoryView &trajectory, std::size_t limit,
                                  Measure measure) {
    return with_cost(trajectory, measure,
                     [&](const auto &cost) { return sequence_impl(trajectory, limit, cost); });
}

} // namespace selection::bottom_up

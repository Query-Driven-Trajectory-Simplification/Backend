#include "backend/engine/query/knn.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace query {

namespace {

// Squared instead of std::hypot, which is slow. Fréchet only takes max and min of distances,
// and squaring keeps their order, so the whole computation can stay squared.
double squared_point_dist(double a_x, double a_y, double b_x, double b_y) {
    const double dx = a_x - b_x;
    const double dy = a_y - b_y;
    return dx * dx + dy * dy;
}

double get_squared_frechet_distance(const core::TrajectoryView &query,
                                    const core::TrajectoryView &reference, double max_squared_dist,
                                    std::vector<double> &prev, std::vector<double> &curr) {
    const std::size_t n = query.size();
    const std::size_t m = reference.size();
    if (n == 0 || m == 0) {
        return std::numeric_limits<double>::infinity();
    }

    prev.resize(m);
    curr.resize(m);
    const double inf = std::numeric_limits<double>::infinity();

    // Only cells below the current k-th distance can improve the answer. Track the
    // reachable span in each row instead of evaluating the entire n-by-m grid.
    double value = 0.0;
    std::size_t last = 0;
    for (std::size_t j = 0; j < m; ++j) {
        value = std::max(
            value, squared_point_dist(query.x[0], query.y[0], reference.x[j], reference.y[j]));
        if (value >= max_squared_dist)
            break;
        prev[j] = value;
        last = j + 1;
    }
    if (last == 0)
        return inf;

    std::size_t first = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t next_first = m;
        std::size_t next_last = 0;
        double left = inf;
        for (std::size_t j = first; j < m; ++j) {
            // Beyond the previous row, only a reachable cell on the left can
            // extend the path. j == last can still use the diagonal predecessor.
            if (j > last && left >= max_squared_dist)
                break;
            const double above = j < last ? prev[j] : inf;
            const double diagonal = j > first && j <= last ? prev[j - 1] : inf;
            const double predecessor = std::min({above, diagonal, left});
            value = std::max(predecessor, squared_point_dist(query.x[i], query.y[i], reference.x[j],
                                                             reference.y[j]));
            left = value < max_squared_dist ? value : inf;
            curr[j] = left;
            if (left < max_squared_dist) {
                next_first = std::min(next_first, j);
                next_last = j + 1;
            }
        }
        if (next_last == 0)
            return inf;
        first = next_first;
        last = next_last;
        std::swap(prev, curr);
    }
    return last == m ? prev[m - 1] : inf;
}

double squared_endpoint_lower_bound(const core::TrajectoryView &query,
                                    const core::TrajectoryView &reference) {
    if (query.size() == 0 || reference.size() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    const std::size_t q_last = query.size() - 1;
    const std::size_t r_last = reference.size() - 1;
    return std::max(squared_point_dist(query.x[0], query.y[0], reference.x[0], reference.y[0]),
                    squared_point_dist(query.x[q_last], query.y[q_last], reference.x[r_last],
                                       reference.y[r_last]));
}

// Point distance to bounding rectangle of a given trajectory bounding box
double squared_dist_to_box(double x, double y, const core::BoundingBox &box) {
    const double dx = std::max({box.x_min - x, 0.0, x - box.x_max});
    const double dy = std::max({box.y_min - y, 0.0, y - box.y_max});
    return dx * dx + dy * dy;
}

double squared_box_extent_lower_bound(const core::TrajectoryView &query,
                                      const core::TrajectoryView &reference) {
    if (query.size() == 0 || reference.size() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    const core::BoundingBox &a = query.bbox;
    const core::BoundingBox &b = reference.bbox;
    // Every extreme coordinate in either trajectory must be matched to a point
    // in the other. This remains useful even when their boxes overlap.
    const double dx = std::max(std::abs(a.x_min - b.x_min), std::abs(a.x_max - b.x_max));
    const double dy = std::max(std::abs(a.y_min - b.y_min), std::abs(a.y_max - b.y_max));
    if (!std::isfinite(dx) || !std::isfinite(dy))
        return 0.0; // Views without a cached box use unbounded defaults.
    return std::max(dx * dx, dy * dy);
}

double squared_points_to_box_lower_bound(const core::TrajectoryView &query,
                                         const core::TrajectoryView &reference, double cutoff) {
    if (query.size() == 0 || reference.size() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    double bound = 0.0;
    for (std::size_t i = 0; i < query.size(); ++i) {
        bound = std::max(bound, squared_dist_to_box(query.x[i], query.y[i], reference.bbox));
        if (bound >= cutoff)
            return bound;
    }
    for (std::size_t j = 0; j < reference.size(); ++j) {
        bound = std::max(bound, squared_dist_to_box(reference.x[j], reference.y[j], query.bbox));
        if (bound >= cutoff)
            return bound;
    }
    return bound;
}

// A greedy monotone path gives an upper bound, not an approximate answer. It
// supplies a finite pruning threshold before the exact top-k heap has filled.
double squared_greedy_upper_bound(const core::TrajectoryView &a, const core::TrajectoryView &b) {
    if (a.size() == 0 || b.size() == 0)
        return std::numeric_limits<double>::infinity();
    const double inf = std::numeric_limits<double>::infinity();
    std::size_t i = 0, j = 0;
    double bound = squared_point_dist(a.x[0], a.y[0], b.x[0], b.y[0]);
    while (i + 1 < a.size() || j + 1 < b.size()) {
        const double diagonal =
            i + 1 < a.size() && j + 1 < b.size()
                ? squared_point_dist(a.x[i + 1], a.y[i + 1], b.x[j + 1], b.y[j + 1])
                : inf;
        const double down =
            i + 1 < a.size() ? squared_point_dist(a.x[i + 1], a.y[i + 1], b.x[j], b.y[j]) : inf;
        const double right =
            j + 1 < b.size() ? squared_point_dist(a.x[i], a.y[i], b.x[j + 1], b.y[j + 1]) : inf;
        if (diagonal <= down && diagonal <= right) {
            ++i;
            ++j;
            bound = std::max(bound, diagonal);
        } else if (down <= right) {
            ++i;
            bound = std::max(bound, down);
        } else {
            ++j;
            bound = std::max(bound, right);
        }
    }
    return bound;
}

struct Candidate {
    double squared_lower_bound;
    std::size_t idx;
};

} // namespace

struct neighbor {
    std::size_t idx;
    double squared_frechet_distance;

    bool operator<(const neighbor &other) const {
        return std::tie(squared_frechet_distance, idx) <
               std::tie(other.squared_frechet_distance, other.idx);
    }
};

std::vector<std::size_t> knn(const core::TrajectoryDb &trajectory_db,
                             const core::TrajectoryView &query_trajectory, std::size_t k) {
    if (k == 0) {
        throw std::invalid_argument("knn: k = 0, no neighbors");
    }

    // Lower bound on the Fréchet distance from the query to each trajectory, sorted ascending.
    // Once a bound reaches the current k-th best distance, every remaining candidate is out too.
    std::vector<Candidate> candidates(trajectory_db.size());
    for (std::size_t i = 0; i < trajectory_db.size(); ++i) {
        const core::TrajectoryView reference = trajectory_db.trajectory(i);
        candidates[i] = {squared_endpoint_lower_bound(query_trajectory, reference), i};
    }
    std::ranges::sort(candidates, [](const Candidate &a, const Candidate &b) {
        return std::tie(a.squared_lower_bound, a.idx) < std::tie(b.squared_lower_bound, b.idx);
    });

    double initial_cutoff = std::numeric_limits<double>::infinity();
    if (k <= candidates.size()) {
        constexpr std::size_t kInitialBoundCandidates = 64;
        const auto count = std::min(candidates.size(), std::max(k, kInitialBoundCandidates));
        std::priority_queue<double> upper_bounds;
        for (std::size_t i = 0; i < count; ++i) {
            const double upper = squared_greedy_upper_bound(
                query_trajectory, trajectory_db.trajectory(candidates[i].idx));
            if (upper_bounds.size() < k)
                upper_bounds.push(upper);
            else if (upper < upper_bounds.top()) {
                upper_bounds.pop();
                upper_bounds.push(upper);
            }
        }
        // Include candidates equal to the upper bound, since we still need k results.
        initial_cutoff =
            std::nextafter(upper_bounds.top(), std::numeric_limits<double>::infinity());
    }

    std::priority_queue<neighbor> curr_neighbors;
    std::vector<double> prev;
    std::vector<double> curr;
    for (const Candidate &candidate : candidates) {
        const std::size_t i = candidate.idx;

        // The greedy paths bound the k-th distance until exact results tighten it.
        const double curr_max_dist =
            curr_neighbors.size() < k
                ? initial_cutoff
                : std::nextafter(curr_neighbors.top().squared_frechet_distance,
                                 std::numeric_limits<double>::infinity());
        if (candidate.squared_lower_bound >= curr_max_dist) {
            break;
        }

        core::TrajectoryView reference_trajectory = trajectory_db.trajectory(i);
        if (std::isfinite(curr_max_dist) &&
            (squared_box_extent_lower_bound(query_trajectory, reference_trajectory) >=
                 curr_max_dist ||
             squared_points_to_box_lower_bound(query_trajectory, reference_trajectory,
                                               curr_max_dist) >= curr_max_dist)) {
            continue;
        }
        double curr_dist = get_squared_frechet_distance(query_trajectory, reference_trajectory,
                                                        curr_max_dist, prev, curr);
        if (curr_dist >= curr_max_dist && std::isfinite(curr_max_dist))
            continue;
        if (curr_neighbors.size() < k) {
            curr_neighbors.push({.idx = i, .squared_frechet_distance = curr_dist});
            continue;
        }
        if (neighbor{.idx = i, .squared_frechet_distance = curr_dist} < curr_neighbors.top()) {
            curr_neighbors.pop();
            curr_neighbors.push({.idx = i, .squared_frechet_distance = curr_dist});
        }
    }

    std::vector<std::size_t> neighbor_ids(curr_neighbors.size());
    for (std::size_t i = neighbor_ids.size(); i > 0; --i) {
        neighbor_ids[i - 1] = curr_neighbors.top().idx;
        curr_neighbors.pop();
    }
    return neighbor_ids;
}
KnnWorkload generate_knn_queries(const core::TrajectoryDb &original,
                                 const core::TrajectoryDb &query_pool, const KnnConfig &config,
                                 std::uint64_t seed) {

    if (config.k == 0) {
        throw std::invalid_argument("knn: k = 0, no neighbors");
    }
    if (config.p < 0.0 || config.p > 1.0) {
        throw std::invalid_argument("knn: p must be in [0, 1]");
    }

    std::vector<std::size_t> picked(query_pool.size());
    std::ranges::iota(picked.begin(), picked.end(), 0);
    if (config.query_count != 0 && config.query_count < picked.size()) {
        std::mt19937_64 rng(seed);
        std::ranges::shuffle(picked, rng);
        picked.resize(config.query_count);
    }

    KnnWorkload results{core::subset(query_pool, picked), {}, config.k, config.p};
    results.entries.resize(results.queries.size());

    core::parallel_for(results.queries.size(), [&](std::size_t i) {
        results.entries[i] = {i, knn(original, results.queries.trajectory(i), config.k)};
    });
    return results;
}

} // namespace query

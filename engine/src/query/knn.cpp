#include "backend/engine/query/knn.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <algorithm>
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
    for (std::size_t i = 0; i < n; ++i) {
        double row_min = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < m; ++j) {
            const double d =
                squared_point_dist(query.x[i], query.y[i], reference.x[j], reference.y[j]);
            if (i == 0 && j == 0) {
                curr[j] = d;
            } else if (i == 0) {
                curr[j] = std::max(d, curr[j - 1]);
            } else if (j == 0) {
                curr[j] = std::max(d, prev[j]);
            } else {
                curr[j] = std::max(d, std::min({prev[j], curr[j - 1], prev[j - 1]}));
            }
            row_min = std::min(row_min, curr[j]);
        }
        // Every path to the end passes through this row, so the result is at least row_min.
        if (row_min >= max_squared_dist) {
            return row_min;
        }
        std::swap(prev, curr);
    }
    return prev[m - 1];
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

double squared_box_gap_lower_bound(const core::TrajectoryView &query,
                                   const core::TrajectoryView &reference) {
    if (query.size() == 0 || reference.size() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    const core::BoundingBox &a = query.bbox;
    const core::BoundingBox &b = reference.bbox;
    const double dx = std::max({a.x_min - b.x_max, 0.0, b.x_min - a.x_max});
    const double dy = std::max({a.y_min - b.y_max, 0.0, b.y_min - a.y_max});
    return dx * dx + dy * dy;
}

double squared_points_to_box_lower_bound(const core::TrajectoryView &query,
                                         const core::TrajectoryView &reference) {
    if (query.size() == 0 || reference.size() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    double bound = 0.0;
    for (std::size_t i = 0; i < query.size(); ++i) {
        bound = std::max(bound, squared_dist_to_box(query.x[i], query.y[i], reference.bbox));
    }
    for (std::size_t j = 0; j < reference.size(); ++j) {
        bound = std::max(bound, squared_dist_to_box(reference.x[j], reference.y[j], query.bbox));
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
        return squared_frechet_distance < other.squared_frechet_distance;
    }
};

std::vector<std::size_t> knn(const core::TrajectoryDb &trajectory_db,
                             const core::TrajectoryView &query_trajectory, std::size_t k) {
    // Lower bound on the Fréchet distance from the query to each trajectory, sorted ascending.
    // Once a bound reaches the current k-th best distance, every remaining candidate is out too.
    std::vector<Candidate> candidates(trajectory_db.size());
    for (std::size_t i = 0; i < trajectory_db.size(); ++i) {
        const core::TrajectoryView reference = trajectory_db.trajectory(i);
        candidates[i] = {std::max(squared_endpoint_lower_bound(query_trajectory, reference),
                                  squared_box_gap_lower_bound(query_trajectory, reference)),
                         i};
    }
    std::ranges::sort(candidates, [](const Candidate &a, const Candidate &b) {
        return std::tie(a.squared_lower_bound, a.idx) < std::tie(b.squared_lower_bound, b.idx);
    });

    std::priority_queue<neighbor> curr_neighbors;
    std::vector<double> prev;
    std::vector<double> curr;
    for (const Candidate &candidate : candidates) {
        const std::size_t i = candidate.idx;

        // Until the heap is full any distance qualifies, so there is no bound to stop early at.
        const double curr_max_dist = curr_neighbors.size() < k
                                         ? std::numeric_limits<double>::infinity()
                                         : curr_neighbors.top().squared_frechet_distance;
        if (candidate.squared_lower_bound >= curr_max_dist) {
            break;
        }

        core::TrajectoryView reference_trajectory = trajectory_db.trajectory(i);
        if (curr_neighbors.size() == k &&
            squared_points_to_box_lower_bound(query_trajectory, reference_trajectory) >=
                curr_max_dist) {
            continue;
        }
        double curr_dist = get_squared_frechet_distance(query_trajectory, reference_trajectory,
                                                        curr_max_dist, prev, curr);
        if (curr_neighbors.size() < k) {
            curr_neighbors.push({.idx = i, .squared_frechet_distance = curr_dist});
            continue;
        }
        if (curr_dist < curr_max_dist) {
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
                                 const core::TrajectoryDb &query_pool, const KnnConfig &config) {

    if (config.k == 0) {
        throw std::invalid_argument("knn: k = 0, no neighbors");
    }

    std::vector<std::size_t> picked(query_pool.size());
    std::ranges::iota(picked.begin(), picked.end(), 0);
    if (config.query_count != 0 && config.query_count < picked.size()) {
        std::mt19937_64 rng(config.seed);
        std::ranges::shuffle(picked, rng);
        picked.resize(config.query_count);
    }

    KnnWorkload results{core::subset(query_pool, picked), {}, config.k};
    results.entries.resize(results.queries.size());

    core::parallel_for(results.queries.size(), [&](std::size_t i) {
        results.entries[i] = {i, knn(original, results.queries.trajectory(i), config.k)};
    });
    return results;
}

} // namespace query

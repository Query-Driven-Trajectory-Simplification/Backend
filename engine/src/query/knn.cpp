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
    if (k == 0) {
        throw std::invalid_argument("knn: k = 0, no neighbors");
    }

    std::priority_queue<neighbor> curr_neighbors;
    std::vector<double> prev;
    std::vector<double> curr;
    for (std::size_t i = 0; i < trajectory_db.size(); ++i) {

        core::TrajectoryView reference_trajectory = trajectory_db.trajectory(i);

        // Until the heap is full any distance qualifies, so there is no bound to stop early at.
        const double curr_max_dist = curr_neighbors.size() < k
                                         ? std::numeric_limits<double>::infinity()
                                         : curr_neighbors.top().squared_frechet_distance;
        // Even the best case for this trajectory cannot beat the current k-th neighbour.
        // shortest leash can never be shorter than first or last point
        if (curr_neighbors.size() == k &&
            squared_endpoint_lower_bound(query_trajectory, reference_trajectory) >= curr_max_dist) {
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
                                 const core::TrajectoryDb &query_pool, const KnnConfig &config,
                                 std::uint64_t seed) {

    if (config.k == 0) {
        throw std::invalid_argument("knn: k = 0, no neighbors");
    }

    std::vector<std::size_t> picked(query_pool.size());
    std::iota(picked.begin(), picked.end(), 0);
    if (config.query_count != 0 && config.query_count < picked.size()) {
        std::mt19937_64 rng(seed);
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

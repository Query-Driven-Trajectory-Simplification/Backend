#include "backend/engine/query/knn.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace query {

namespace {

double point_dist(double a_x, double a_y, double b_x, double b_y){
    return std::hypot((a_x-b_x), (a_y-b_y));

}
// Discrete Fréchet distance between query and reference. Returns early with a value >= max_dist
// as soon as the distance is known to be at least max_dist. Infinite if either is empty.
double get_frechet_distance(const core::TrajectoryView &query,
                            const core::TrajectoryView &reference, double max_dist) {
    const std::size_t n = query.size();
    const std::size_t m = reference.size();
    if (n == 0 || m == 0) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<double> prev(m);
    std::vector<double> curr(m);
    for (std::size_t i = 0; i < n; ++i) {
        double row_min = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < m; ++j) {
            const double d =
                point_dist(query.x[i], query.y[i], reference.x[j], reference.y[j]);
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
        if (row_min >= max_dist) {
            return row_min;
        }
        std::swap(prev, curr);
    }
    return prev[m - 1];
}
} // namespace

struct neighbor {
    std::size_t idx;
    double frechet_distance;

    bool operator<(const neighbor &other) const {
        return frechet_distance < other.frechet_distance;
    }
};

std::vector<std::size_t> knn(const core::TrajectoryDb &trajectory_db,
                             const core::TrajectoryView &query_trajectory, std::size_t k) {

    std::priority_queue<neighbor> curr_neighbors;
    for (std::size_t i = 0; i < trajectory_db.size(); ++i) {

        core::TrajectoryView reference_trajectory = trajectory_db.trajectory(i);

        // Until the heap is full any distance qualifies, so there is no bound to stop early at.
        const double curr_max_dist = curr_neighbors.size() < k
                                         ? std::numeric_limits<double>::infinity()
                                         : curr_neighbors.top().frechet_distance;
        double curr_dist =
            get_frechet_distance(query_trajectory, reference_trajectory, curr_max_dist);
        if (curr_neighbors.size() < k) {
            curr_neighbors.push({.idx = i, .frechet_distance = curr_dist});
            continue;
        }
        if (curr_dist < curr_max_dist) {
            curr_neighbors.pop();
            curr_neighbors.push({.idx = i, .frechet_distance = curr_dist});
        }
    }

    std::vector<std::size_t> neighbor_ids(curr_neighbors.size());
    for (std::size_t i = neighbor_ids.size(); i > 0; --i) {
        neighbor_ids[i - 1] = curr_neighbors.top().idx;
        curr_neighbors.pop();
    }
    return neighbor_ids;
}

core::TrajectoryDb generate_knn_queries(const core::TrajectoryDb & /*trajectories*/) {
    throw std::logic_error("query::generate_knn_queries: query generation not implemented yet");
}

} // namespace query

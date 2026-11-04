#include "backend/engine/query/frechet_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace query {

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

    // Cells at or above the cutoff keep their value instead of becoming infinity. They can only
    // lead to values at or above the cutoff, so every value below it stays exact.
    std::size_t first = 0;
    for (std::size_t i = 1; i < n; ++i) {
        const double qx = query.x[i];
        const double qy = query.y[i];
        const auto dist = [&](std::size_t j) {
            return squared_point_dist(qx, qy, reference.x[j], reference.y[j]);
        };
        std::size_t next_first = m;
        std::size_t next_last = 0;

        // Within the previous row's reachable span [first, last), the cell above always exists,
        // and the diagonal does for every cell but the first
        double left = std::max(prev[first], dist(first));
        curr[first] = left;
        if (left < max_squared_dist) {
            next_first = first;
            next_last = first + 1;
        }
        for (std::size_t j = first + 1; j < last; ++j) {
            left = std::max(std::min({prev[j], prev[j - 1], left}), dist(j));
            curr[j] = left;
            if (left < max_squared_dist) {
                next_first = std::min(next_first, j);
                next_last = j + 1;
            }
        }

        // Past the span only the cell on the left continues a path, plus the diagonal at j == last
        for (std::size_t j = last; j < m; ++j) {
            const double predecessor = j == last ? std::min(prev[last - 1], left) : left;
            if (predecessor >= max_squared_dist)
                break;
            left = std::max(predecessor, dist(j));
            if (left >= max_squared_dist)
                break;
            curr[j] = left;
            next_first = std::min(next_first, j);
            next_last = j + 1;
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

double frechet_distance(const core::TrajectoryView &a, const core::TrajectoryView &b) {
    std::vector<double> prev;
    std::vector<double> curr;
    return std::sqrt(get_squared_frechet_distance(a, b, std::numeric_limits<double>::infinity(),
                                                  prev, curr));
}

bool frechet_within(const core::TrajectoryView &a, const core::TrajectoryView &b,
                    double max_dist) {
    // Values below the cutoff are exact, so anything at most max_dist^2 is found. The relative
    // slack absorbs rounding when max_dist is itself a distance, as sqrt(d)^2 can fall below d.
    const double cutoff = std::nextafter(max_dist * max_dist * (1.0 + 1e-12),
                                         std::numeric_limits<double>::infinity());
    std::vector<double> prev;
    std::vector<double> curr;
    return get_squared_frechet_distance(a, b, cutoff, prev, curr) < cutoff;
}

double endpoint_lower_bound(const core::TrajectoryView &a, const core::TrajectoryView &b) {
    return std::sqrt(squared_endpoint_lower_bound(a, b));
}

} // namespace query

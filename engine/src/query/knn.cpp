#include "backend/engine/query/knn.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/frechet_helper.hpp"
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

// Point distance to bounding rectangle of a given trajectory bounding box
double squared_dist_to_box(double x, double y, const core::BoundingBox &box) {
    const double dx = std::max({box.x_min - x, 0.0, x - box.x_max});
    const double dy = std::max({box.y_min - y, 0.0, y - box.y_max});
    return dx * dx + dy * dy;
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

double squared_segment_dist(double px, double py, double ax, double ay, double bx, double by) {
    // Endpoints are measured directly and interior projections by the cross product, so a point
    // on the segment is not pushed off it by reconstructing a + t (b - a)
    const double dx = bx - ax, dy = by - ay;
    const double length = dx * dx + dy * dy;
    const double along = (px - ax) * dx + (py - ay) * dy;
    if (length == 0.0 || along <= 0.0)
        return squared_point_dist(px, py, ax, ay);
    if (along >= length)
        return squared_point_dist(px, py, bx, by);
    const double cross = (px - ax) * dy - (py - ay) * dx;
    return cross * cross / length;
}

// Largest distance from a vertex of a to the polyline b, a lower bound on the continuous
// distance. A vertex stops scanning once it is no farther than the current maximum, and the
// whole scan stops above stop.
double squared_directed_hausdorff(const core::TrajectoryView &a, const core::TrajectoryView &b,
                                  double stop) {
    const double inf = std::numeric_limits<double>::infinity();
    if (a.size() == 0 || b.size() == 0)
        return inf;
    double bound = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double nearest = b.size() == 1 ? squared_point_dist(a.x[i], a.y[i], b.x[0], b.y[0]) : inf;
        for (std::size_t j = 0; j + 1 < b.size() && nearest > bound; ++j)
            nearest = std::min(nearest, squared_segment_dist(a.x[i], a.y[i], b.x[j], b.y[j],
                                                             b.x[j + 1], b.y[j + 1]));
        bound = std::max(bound, nearest);
        if (bound > stop)
            return bound;
    }
    return bound;
}

// Candidates ordered by the stronger of the endpoint and box-extent bounds. Once a bound
// reaches the current k-th best distance, every remaining candidate is out too.
class CandidateQueue {
  public:
    CandidateQueue(const core::TrajectoryDb &trajectory_db, const core::TrajectoryView &query,
                   std::size_t k) {
        m_heap.resize(trajectory_db.size());
        for (std::size_t i = 0; i < trajectory_db.size(); ++i) {
            const core::TrajectoryView reference = trajectory_db.trajectory(i);
            m_heap[i] = {std::max(squared_endpoint_lower_bound(query, reference),
                                  squared_box_extent_lower_bound(query, reference)),
                         i};
        }
        std::make_heap(m_heap.begin(), m_heap.end(), farther);
        if (k > m_heap.size())
            return;
        // The greedy paths bound the k-th distance until exact results tighten it
        constexpr std::size_t kInitialBoundCandidates = 64;
        const auto count = std::min(m_heap.size(), std::max(k, kInitialBoundCandidates));
        m_seeds.reserve(count);
        std::priority_queue<double> upper_bounds;
        for (std::size_t i = 0; i < count; ++i) {
            m_seeds.push_back(extract());
            const double upper =
                squared_greedy_upper_bound(query, trajectory_db.trajectory(m_seeds.back().idx));
            if (upper_bounds.size() < k)
                upper_bounds.push(upper);
            else if (upper < upper_bounds.top()) {
                upper_bounds.pop();
                upper_bounds.push(upper);
            }
        }
        m_initial_cutoff = upper_bounds.top();
    }

    bool empty() const { return m_next_seed == m_seeds.size() && m_heap.empty(); }
    Candidate next() { return m_next_seed < m_seeds.size() ? m_seeds[m_next_seed++] : extract(); }
    // Squared distance the k-th neighbour is known not to exceed, infinity without k candidates
    double initial_cutoff() const { return m_initial_cutoff; }

  private:
    static bool farther(const Candidate &a, const Candidate &b) {
        return std::tie(a.squared_lower_bound, a.idx) > std::tie(b.squared_lower_bound, b.idx);
    }
    Candidate extract() {
        std::pop_heap(m_heap.begin(), m_heap.end(), farther);
        const Candidate candidate = m_heap.back();
        m_heap.pop_back();
        return candidate;
    }

    std::vector<Candidate> m_heap;
    std::vector<Candidate> m_seeds;
    std::size_t m_next_seed = 0;
    double m_initial_cutoff = std::numeric_limits<double>::infinity();
};

struct neighbor {
    std::size_t idx;
    double squared_frechet_distance;

    bool operator<(const neighbor &other) const {
        return std::tie(squared_frechet_distance, idx) <
               std::tie(other.squared_frechet_distance, other.idx);
    }
};

std::vector<std::pair<std::size_t, double>>
discrete_knn(const core::TrajectoryDb &trajectory_db, const core::TrajectoryView &query_trajectory,
             std::size_t k) {
    CandidateQueue queue(trajectory_db, query_trajectory, k);
    // Include candidates equal to the upper bound, since we still need k results
    const double initial_cutoff =
        std::nextafter(queue.initial_cutoff(), std::numeric_limits<double>::infinity());
    std::priority_queue<neighbor> curr_neighbors;
    std::vector<double> prev;
    std::vector<double> curr;
    while (!queue.empty()) {
        const Candidate candidate = queue.next();
        const std::size_t i = candidate.idx;
        const double curr_max_dist =
            curr_neighbors.size() < k
                ? initial_cutoff
                : std::nextafter(curr_neighbors.top().squared_frechet_distance,
                                 std::numeric_limits<double>::infinity());
        if (candidate.squared_lower_bound >= curr_max_dist)
            break;

        core::TrajectoryView reference_trajectory = trajectory_db.trajectory(i);
        if (std::isfinite(curr_max_dist) &&
            squared_points_to_box_lower_bound(query_trajectory, reference_trajectory,
                                              curr_max_dist) >= curr_max_dist)
            continue;
        const double curr_dist = get_squared_frechet_distance(
            query_trajectory, reference_trajectory, curr_max_dist, prev, curr);
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

    std::vector<std::pair<std::size_t, double>> result(curr_neighbors.size());
    for (std::size_t i = result.size(); i > 0; --i) {
        result[i - 1] = {curr_neighbors.top().idx,
                         std::sqrt(curr_neighbors.top().squared_frechet_distance)};
        curr_neighbors.pop();
    }
    return result;
}

// The first millimetre radius accepted by the continuous decision predicate. Unlike an
// arbitrary bisection midpoint, this key is identical for different valid initial bounds.
double continuous_rank_distance(const core::TrajectoryView &a, const core::TrajectoryView &b,
                                double lower, double upper) {
    if (!std::isfinite(upper))
        return std::numeric_limits<double>::infinity();
    // Geometric lower bounds and the decision predicate have rounding slack. Start below it,
    // so an accepted grid point is never skipped just because a bound rounded upward.
    const double slack = 2e-12 * std::max(1.0, lower);
    const double first = std::ceil(std::max(0.0, lower - slack) / kKnnResolutionM);
    const double last = std::ceil(upper / kKnnResolutionM);
    if (last >= static_cast<double>(std::numeric_limits<std::uint64_t>::max()))
        throw std::overflow_error("knn: distance exceeds millimetre ranking range");
    auto lo = static_cast<std::uint64_t>(first);
    auto hi = static_cast<std::uint64_t>(last);
    // Most distances equal their Hausdorff bound. Resolve those with one decision instead of
    // bisecting the entire interval up to the discrete upper bound.
    if (lo < hi && continuous_frechet_within(a, b, static_cast<double>(lo) * kKnnResolutionM))
        return static_cast<double>(lo) * kKnnResolutionM;
    if (lo < hi)
        ++lo;
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        if (continuous_frechet_within(a, b, static_cast<double>(mid) * kKnnResolutionM))
            hi = mid;
        else
            lo = mid + 1;
    }
    return static_cast<double>(lo) * kKnnResolutionM;
}

struct RankedNeighbor {
    double distance;
    std::size_t idx;
    bool operator<(const RankedNeighbor &other) const {
        return knn_rank_less(distance, idx, other.distance, other.idx);
    }
};

std::vector<std::pair<std::size_t, double>> continuous_knn(const core::TrajectoryDb &trajectory_db,
                                                           const core::TrajectoryView &query,
                                                           std::size_t k,
                                                           const KnownDistance &known) {
    CandidateQueue queue(trajectory_db, query, k);
    std::priority_queue<RankedNeighbor> nearest;
    std::vector<double> prev, curr;
    auto offer = [&](const RankedNeighbor &next) {
        if (nearest.size() < k)
            nearest.push(next);
        else if (next < nearest.top()) {
            nearest.pop();
            nearest.push(next);
        }
    };
    while (!queue.empty()) {
        const Candidate candidate = queue.next();
        // One extra bin absorbs floating-point rounding in bounds and includes all distance ties.
        const double limit =
            (nearest.size() < k ? std::sqrt(queue.initial_cutoff()) : nearest.top().distance) +
            kKnnResolutionM;
        const double squared_limit = limit * limit;
        if (candidate.squared_lower_bound > squared_limit)
            break;
        // Found from the other side: no Fréchet computation needed
        if (known) {
            if (const auto distance = known(candidate.idx)) {
                if (std::isfinite(*distance) && *distance <= limit)
                    offer({*distance, candidate.idx});
                continue;
            }
        }
        const auto reference = trajectory_db.trajectory(candidate.idx);
        if (squared_points_to_box_lower_bound(query, reference, squared_limit) > squared_limit)
            continue;
        const double forward = squared_directed_hausdorff(query, reference, squared_limit);
        if (forward > squared_limit)
            continue;
        const double squared_hausdorff =
            std::max(forward, squared_directed_hausdorff(reference, query, squared_limit));
        if (squared_hausdorff > squared_limit)
            continue;
        double upper = limit;
        const double discrete = get_squared_frechet_distance(
            query, reference,
            std::nextafter(squared_limit, std::numeric_limits<double>::infinity()), prev, curr);
        if (discrete <= squared_limit)
            upper = std::min(upper, std::sqrt(discrete));
        else if (!continuous_frechet_within(query, reference, limit))
            continue;
        const double lower =
            std::min(upper, std::sqrt(std::max(candidate.squared_lower_bound, squared_hausdorff)));
        const RankedNeighbor next{continuous_rank_distance(query, reference, lower, upper),
                                  candidate.idx};
        if (!std::isfinite(next.distance))
            continue;
        offer(next);
    }
    std::vector<std::pair<std::size_t, double>> result(nearest.size());
    for (std::size_t i = result.size(); i > 0; --i) {
        result[i - 1] = {nearest.top().idx, nearest.top().distance};
        nearest.pop();
    }
    return result;
}

} // namespace

double knn_distance(const core::TrajectoryView &a, const core::TrajectoryView &b,
                    FrechetVariant variant) {
    const double upper = frechet_distance(a, b);
    if (variant == FrechetVariant::Discrete || !std::isfinite(upper))
        return upper;
    const double squared_upper = upper * upper;
    const double lower = std::sqrt(
        std::max({squared_endpoint_lower_bound(a, b), squared_box_extent_lower_bound(a, b),
                  squared_directed_hausdorff(a, b, squared_upper),
                  squared_directed_hausdorff(b, a, squared_upper)}));
    return continuous_rank_distance(a, b, std::min(lower, upper), upper);
}

std::vector<std::size_t> knn(const core::TrajectoryDb &trajectory_db,
                             const core::TrajectoryView &query_trajectory, std::size_t k,
                             FrechetVariant variant) {
    if (k == 0) {
        throw std::invalid_argument("knn: k = 0, no neighbors");
    }
    const auto ranked = knn_with_distances(trajectory_db, query_trajectory, k, variant);
    std::vector<std::size_t> result(ranked.size());
    for (std::size_t i = 0; i < ranked.size(); ++i)
        result[i] = ranked[i].first;
    return result;
}

std::vector<std::pair<std::size_t, double>>
knn_with_distances(const core::TrajectoryDb &trajectory_db,
                   const core::TrajectoryView &query_trajectory, std::size_t k,
                   FrechetVariant variant, const KnownDistance &known) {
    if (k == 0) {
        throw std::invalid_argument("knn: k = 0, no neighbors");
    }
    return variant == FrechetVariant::Discrete
               ? discrete_knn(trajectory_db, query_trajectory, k)
               : continuous_knn(trajectory_db, query_trajectory, k, known);
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
    std::iota(picked.begin(), picked.end(), 0);
    if (config.query_count != 0 && config.query_count < picked.size()) {
        std::mt19937_64 rng(seed);
        std::ranges::shuffle(picked, rng);
        picked.resize(config.query_count);
    }

    KnnWorkload results{core::subset(query_pool, picked), {}, config.k, config.p, config.frechet};
    results.entries.resize(results.queries.size());

    core::parallel_for(results.queries.size(), [&](std::size_t i) {
        results.entries[i] = {
            i, knn(original, results.queries.trajectory(i), config.k, config.frechet)};
    });
    return results;
}

} // namespace query

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

std::vector<std::size_t> discrete_knn(const core::TrajectoryDb &trajectory_db,
                                      const core::TrajectoryView &query_trajectory, std::size_t k) {
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

    std::vector<std::size_t> neighbor_ids(curr_neighbors.size());
    for (std::size_t i = neighbor_ids.size(); i > 0; --i) {
        neighbor_ids[i - 1] = curr_neighbors.top().idx;
        curr_neighbors.pop();
    }
    return neighbor_ids;
}

// A candidate whose continuous distance is known to lie in [lo, hi]
struct Bracket {
    std::size_t idx;
    double lo;
    double hi;
};

// Each distance is narrowed only as far as top-k membership and order need. Distances within
// tolerance of each other count as equal and are ordered by database index, and the same rule
// decides membership at the k-th place. Every prune allows the tolerance as well.
std::vector<std::size_t> continuous_knn(const core::TrajectoryDb &trajectory_db,
                                        const core::TrajectoryView &query, std::size_t k) {
    constexpr double tolerance = 1e-3;
    CandidateQueue queue(trajectory_db, query, k);
    std::vector<Bracket> kept;
    std::vector<double> scratch;
    const auto halve = [&](Bracket &b) {
        const double mid = 0.5 * (b.lo + b.hi);
        if (continuous_frechet_within(query, trajectory_db.trajectory(b.idx), mid))
            b.hi = mid;
        else
            b.lo = mid;
    };
    const auto kth_upper = [&] {
        scratch.clear();
        for (const auto &b : kept)
            scratch.push_back(b.hi);
        std::nth_element(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(k - 1),
                         scratch.end());
        return scratch[k - 1];
    };
    // Narrows overlapping brackets until they are disjoint or within tolerance, then orders runs
    // of overlapping brackets, which are ties, by index
    const auto rank = [&] {
        const auto by_lower = [](const Bracket &b) { return std::pair(b.lo, b.idx); };
        for (bool overlapping = true; overlapping;) {
            overlapping = false;
            std::ranges::sort(kept, {}, by_lower);
            for (std::size_t i = 0; i + 1 < kept.size(); ++i) {
                Bracket &a = kept[i], &b = kept[i + 1];
                if (a.hi < b.lo)
                    continue;
                Bracket &wider = a.hi - a.lo >= b.hi - b.lo ? a : b;
                if (wider.hi - wider.lo <= tolerance)
                    continue;
                halve(wider);
                overlapping = true;
            }
        }
        for (std::size_t start = 0; start < kept.size();) {
            std::size_t end = start + 1;
            double reach = kept[start].hi;
            while (end < kept.size() && kept[end].lo <= reach)
                reach = std::max(reach, kept[end++].hi);
            std::ranges::sort(kept.begin() + static_cast<std::ptrdiff_t>(start),
                              kept.begin() + static_cast<std::ptrdiff_t>(end), {}, &Bracket::idx);
            start = end;
        }
    };
    // Drops brackets that are certainly beyond the k-th, refining the widest uncertain one.
    // Once only ties remain at the boundary, the ranking decides who stays.
    const auto trim = [&] {
        while (kept.size() > k) {
            const double cutoff = kth_upper();
            std::erase_if(kept, [&](const Bracket &b) { return b.lo > cutoff + tolerance; });
            if (kept.size() <= k)
                return;
            Bracket *widest = nullptr;
            for (auto &b : kept)
                if (b.hi >= cutoff && b.hi - b.lo > tolerance &&
                    (!widest || b.hi - b.lo > widest->hi - widest->lo))
                    widest = &b;
            if (!widest) {
                rank();
                kept.resize(k);
                return;
            }
            halve(*widest);
        }
    };

    std::vector<double> prev, curr;
    while (!queue.empty()) {
        const Candidate candidate = queue.next();
        const double cutoff = kept.size() < k ? std::sqrt(queue.initial_cutoff()) : kth_upper();
        const double limit = cutoff + tolerance;
        const double squared_limit = limit * limit;
        if (candidate.squared_lower_bound > squared_limit)
            break;
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
        // The discrete distance bounds the continuous one from above
        double hi = limit;
        const double squared_discrete = get_squared_frechet_distance(
            query, reference, std::nextafter(squared_limit, squared_limit + 1.0), prev, curr);
        if (squared_discrete <= squared_limit)
            hi = std::min(hi, std::sqrt(squared_discrete));
        else if (!continuous_frechet_within(query, reference, limit))
            continue;
        double lo = std::min(
            hi, std::max(std::sqrt(candidate.squared_lower_bound), std::sqrt(squared_hausdorff)));
        // The distance often equals the Hausdorff bound
        if (hi - lo > tolerance) {
            if (continuous_frechet_within(query, reference, lo + tolerance))
                hi = lo + tolerance;
            else
                lo += tolerance;
        }
        kept.push_back({candidate.idx, lo, hi});
        trim();
    }

    rank();
    std::vector<std::size_t> neighbor_ids;
    for (const auto &b : kept)
        neighbor_ids.push_back(b.idx);
    return neighbor_ids;
}

} // namespace

std::vector<std::size_t> knn(const core::TrajectoryDb &trajectory_db,
                             const core::TrajectoryView &query_trajectory, std::size_t k,
                             FrechetVariant variant) {
    if (k == 0) {
        throw std::invalid_argument("knn: k = 0, no neighbors");
    }
    return variant == FrechetVariant::Discrete ? discrete_knn(trajectory_db, query_trajectory, k)
                                               : continuous_knn(trajectory_db, query_trajectory, k);
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

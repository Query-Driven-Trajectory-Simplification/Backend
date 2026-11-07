#include "backend/engine/query/frechet_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace query {

namespace {

struct Interval {
    double lo = 2.0;
    double hi = -1.0;
    bool empty() const { return lo > hi; }
};

// Parameters t in [0, 1] where |a + t(b - a) - p|^2 <= eps2
Interval free_interval(double px, double py, double ax, double ay, double bx, double by,
                       double eps2) {
    const double dx = bx - ax, dy = by - ay;
    const double wx = ax - px, wy = ay - py;
    const double a = dx * dx + dy * dy;
    const double start = wx * wx + wy * wy;
    const double ex = bx - px, ey = by - py;
    const double end = ex * ex + ey * ey;
    const bool start_in = start <= eps2;
    const bool end_in = end <= eps2;
    if (start_in && end_in)
        return {0.0, 1.0}; // the disc is convex
    if (a == 0.0)
        return start_in ? Interval{0.0, 1.0} : Interval{};
    // Closest parameter and squared distance to the line. The cross product avoids the
    // cancellation of the quadratic's discriminant on long segments.
    const double closest = -(wx * dx + wy * dy) / a;
    const double cross = wx * dy - wy * dx;
    const double line2 = cross * cross / a;
    if (line2 > eps2)
        return {};
    const double half = std::sqrt((eps2 - line2) / a);
    const double lo = start_in ? 0.0 : std::max(0.0, closest - half);
    const double hi = end_in ? 1.0 : std::min(1.0, closest + half);
    if (lo > hi)
        return {};
    return {lo, hi};
}

struct Scratch {
    std::vector<Interval> cur, next;
};

// Free-space reachability over the cells of P's segments (columns) against Q's (rows)
bool decide(std::span<const double> px, std::span<const double> py, std::span<const double> qx,
            std::span<const double> qy, double eps, Scratch &s) {
    const std::size_t n = px.size(), m = qx.size();
    if (n == 0 || m == 0)
        return false;
    const double eps2 =
        std::nextafter(eps * eps * (1.0 + 1e-12), std::numeric_limits<double>::infinity());
    const auto d2 = [](double ax, double ay, double bx, double by) {
        const double dx = ax - bx, dy = ay - by;
        return dx * dx + dy * dy;
    };
    if (d2(px[0], py[0], qx[0], qy[0]) > eps2 ||
        d2(px[n - 1], py[n - 1], qx[m - 1], qy[m - 1]) > eps2)
        return false;
    // A single point against a polyline: every vertex must be within eps
    if (n == 1 || m == 1) {
        const auto &ax = n == 1 ? qx : px;
        const auto &ay = n == 1 ? qy : py;
        const double cx = n == 1 ? px[0] : qx[0];
        const double cy = n == 1 ? py[0] : qy[0];
        for (std::size_t k = 0; k < ax.size(); ++k)
            if (d2(ax[k], ay[k], cx, cy) > eps2)
                return false;
        return true;
    }

    const std::size_t segs_q = m - 1;
    s.cur.assign(segs_q, Interval{});
    s.next.assign(segs_q, Interval{});

    // Column 0, left boundary: reachable only as a free prefix from (0, 0)
    std::size_t first = segs_q, last = 0;
    for (std::size_t j = 0; j < segs_q; ++j) {
        const Interval f = free_interval(px[0], py[0], qx[j], qy[j], qx[j + 1], qy[j + 1], eps2);
        if (f.empty() || f.lo != 0.0)
            break;
        s.cur[j] = f;
        first = 0;
        last = j;
        if (f.hi != 1.0)
            break;
    }
    Interval bottom = free_interval(qx[0], qy[0], px[0], py[0], px[1], py[1], eps2);
    if (bottom.lo != 0.0)
        bottom = {};

    Interval top_last{};
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const double ax = px[i], ay = py[i], bx = px[i + 1], by = py[i + 1];
        std::size_t next_first = segs_q, next_last = 0;
        bool any = false;
        std::size_t j = bottom.empty() ? first : 0;
        if (j >= segs_q) {
            if (!bottom.empty())
                j = 0;
            else
                return false;
        }
        Interval below = bottom;
        for (; j < segs_q; ++j) {
            const bool has_left = j >= first && j <= last && !s.cur[j].empty();
            const Interval left = has_left ? s.cur[j] : Interval{};
            if (left.empty() && below.empty()) {
                if (j > last)
                    break;
                continue;
            }
            const double cx = qx[j], cy = qy[j], dx = qx[j + 1], dy = qy[j + 1];
            Interval right = free_interval(bx, by, cx, cy, dx, dy, eps2);
            if (below.empty() && !right.empty())
                right.lo = std::max(right.lo, left.lo);
            Interval top = free_interval(dx, dy, ax, ay, bx, by, eps2);
            if (left.empty() && !top.empty())
                top.lo = std::max(top.lo, below.lo);
            if (right.empty())
                right = {};
            if (top.empty())
                top = {};
            s.next[j] = right;
            if (!right.empty()) {
                if (!any)
                    next_first = j;
                next_last = j;
                any = true;
            }
            below = top;
            if (j + 1 == segs_q)
                top_last = top;
        }
        // Bottom boundary for column i + 1
        Interval next_bottom{};
        if (i + 2 < n && !bottom.empty() && bottom.hi == 1.0) {
            next_bottom = free_interval(qx[0], qy[0], bx, by, px[i + 2], py[i + 2], eps2);
            if (next_bottom.lo != 0.0)
                next_bottom = {};
        }
        if (i + 2 == n) {
            const bool right_corner =
                any && next_last == segs_q - 1 && s.next[segs_q - 1].hi == 1.0;
            const bool top_corner = j >= segs_q && !top_last.empty() && top_last.hi == 1.0;
            return right_corner || top_corner;
        }
        if (!any && next_bottom.empty())
            return false;
        for (std::size_t k = first; k <= last && k < segs_q; ++k)
            s.cur[k] = {};
        std::swap(s.cur, s.next);
        first = any ? next_first : segs_q;
        last = any ? next_last : 0;
        bottom = next_bottom;
        top_last = {};
    }
    return false;
}

} // namespace

bool continuous_frechet_within(const core::TrajectoryView &a, const core::TrajectoryView &b,
                               double max_dist) {
    thread_local Scratch scratch;
    return decide(a.x, a.y, b.x, b.y, max_dist, scratch);
}

double continuous_frechet_distance(const core::TrajectoryView &a, const core::TrajectoryView &b,
                                   double lower, double upper) {
    constexpr double tolerance = 1e-3;
    if (!std::isfinite(upper) || !continuous_frechet_within(a, b, upper))
        return std::numeric_limits<double>::infinity();
    lower = std::max(0.0, std::min(lower, upper));
    while (upper - lower > tolerance) {
        const double mid = 0.5 * (lower + upper);
        if (continuous_frechet_within(a, b, mid))
            upper = mid;
        else
            lower = mid;
    }
    return upper;
}

bool frechet_within(const core::TrajectoryView &a, const core::TrajectoryView &b, double max_dist,
                    FrechetVariant variant) {
    return variant == FrechetVariant::Continuous ? continuous_frechet_within(a, b, max_dist)
                                                 : frechet_within(a, b, max_dist);
}

} // namespace query

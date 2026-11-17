#include "backend/engine/teacher/curves.hpp"

#include <limits>
#include <queue>
#include <stdexcept>

namespace teacher {

namespace {

constexpr std::size_t kMaxStep = 3;

struct Offer {
    double ratio; // gain per point
    std::size_t i, k, version;
    bool operator<(const Offer &o) const {
        return ratio < o.ratio || (ratio == o.ratio && i > o.i);
    }
};

// The best step of 1 to kMaxStep points for trajectory i, at most left points. k = 0 if none
Offer best_offer(const Curve &c, std::size_t i, std::size_t b, std::size_t left,
                 std::size_t version) {
    Offer best{-std::numeric_limits<double>::infinity(), i, 0, version};
    for (std::size_t k = 1; k <= kMaxStep && k <= left && b + k <= c.max_budget(); ++k) {
        const double ratio = (c.at(b + k) - c.at(b)) / static_cast<double>(k);
        if (ratio > best.ratio) {
            best.ratio = ratio;
            best.k = k;
        }
    }
    return best;
}

} // namespace

std::vector<std::size_t> allocate(const std::vector<Curve> &curves, std::size_t total_budget,
                                  double *value) {
    const std::size_t n = curves.size();
    if (total_budget < 2 * n)
        throw std::invalid_argument("allocate: budget below two endpoints per trajectory");
    for (const auto &c : curves)
        if (c.value.empty())
            throw std::invalid_argument("allocate: empty curve");

    std::vector<std::size_t> b(n, 2);
    std::size_t left = total_budget - 2 * n;

    std::vector<std::size_t> version(n, 0);
    std::priority_queue<Offer> queue;
    for (std::size_t i = 0; i < n; ++i)
        if (const auto o = best_offer(curves[i], i, b[i], left, 0); o.k != 0)
            queue.push(o);
    while (left > 0) {
        if (queue.empty())
            throw std::runtime_error("allocate: curves too short to hold the budget");
        const auto o = queue.top();
        queue.pop();
        if (o.version != version[o.i])
            continue;
        if (o.k > left) { // the budget left shrank since this offer was made
            if (const auto again = best_offer(curves[o.i], o.i, b[o.i], left, version[o.i]);
                again.k != 0)
                queue.push(again);
            continue;
        }
        b[o.i] += o.k;
        left -= o.k;
        ++version[o.i];
        if (const auto again = best_offer(curves[o.i], o.i, b[o.i], left, version[o.i]);
            again.k != 0)
            queue.push(again);
    }

    if (value) {
        *value = 0;
        for (std::size_t i = 0; i < n; ++i)
            *value += curves[i].at(b[i]);
    }
    return b;
}

} // namespace teacher

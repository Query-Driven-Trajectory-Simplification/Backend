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
    bool operator<(const Offer &other) const {
        return ratio < other.ratio || (ratio == other.ratio && i > other.i);
    }
};

// The best step of 1 to kMaxStep points for trajectory i, at most left points. k = 0 if none
Offer best_offer(const Curve &curve, std::size_t i, std::size_t budget, std::size_t left,
                 std::size_t version) {
    Offer best{-std::numeric_limits<double>::infinity(), i, 0, version};
    for (std::size_t k = 1; k <= kMaxStep && k <= left && budget + k <= curve.max_budget(); ++k) {
        const double ratio = (curve.at(budget + k) - curve.at(budget)) / static_cast<double>(k);
        if (ratio > best.ratio) {
            best.ratio = ratio;
            best.k = k;
        }
    }
    return best;
}

// Queues trajectory i's best offer from its current budget, if it has one
void push_offer(std::priority_queue<Offer> &queue, const Curve &curve, std::size_t i,
                std::size_t budget, std::size_t left, std::size_t version) {
    if (const auto offer = best_offer(curve, i, budget, left, version); offer.k != 0)
        queue.push(offer);
}

} // namespace

std::vector<std::size_t> allocate(const std::vector<Curve> &curves, std::size_t total_budget,
                                  double *value) {
    const std::size_t n = curves.size();
    if (total_budget < 2 * n)
        throw std::invalid_argument("allocate: budget below two endpoints per trajectory");
    for (const auto &curve : curves)
        if (curve.value.empty())
            throw std::invalid_argument("allocate: empty curve");

    std::vector<std::size_t> budgets(n, 2);
    std::size_t left = total_budget - 2 * n;

    std::vector<std::size_t> version(n, 0);
    std::priority_queue<Offer> queue;
    for (std::size_t i = 0; i < n; ++i)
        push_offer(queue, curves[i], i, budgets[i], left, 0);
    while (left > 0) {
        if (queue.empty())
            throw std::runtime_error("allocate: curves too short to hold the budget");
        const auto offer = queue.top();
        queue.pop();
        if (offer.version != version[offer.i])
            continue;
        // If the budget left shrank since this offer was made, offer again within it
        if (offer.k <= left) {
            budgets[offer.i] += offer.k;
            left -= offer.k;
            ++version[offer.i];
        }
        push_offer(queue, curves[offer.i], offer.i, budgets[offer.i], left, version[offer.i]);
    }

    if (value) {
        *value = 0;
        for (std::size_t i = 0; i < n; ++i)
            *value += curves[i].at(budgets[i]);
    }
    return budgets;
}

} // namespace teacher

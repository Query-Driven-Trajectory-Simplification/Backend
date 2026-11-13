#include "backend/engine/teacher/curves.hpp"

#include <limits>
#include <queue>
#include <stdexcept>

namespace teacher {

namespace {

constexpr std::size_t kMaxStep = 3;
constexpr double kMinGain = 1e-12;
constexpr std::size_t kMaxExchanges = 10'000'000;

struct Offer {
    double ratio; // gain per point
    std::size_t i, k, version;
    bool operator<(const Offer &o) const { return ratio < o.ratio || (ratio == o.ratio && i > o.i); }
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

// The best and second best index by score, so a donor and a receiver can differ
struct Top2 {
    double first = -std::numeric_limits<double>::infinity(), second = first;
    std::size_t first_i = 0, second_i = 0;
    void offer(double score, std::size_t i) {
        if (score > first) {
            second = first;
            second_i = first_i;
            first = score;
            first_i = i;
        } else if (score > second) {
            second = score;
            second_i = i;
        }
    }
};

} // namespace

std::vector<std::size_t> allocate(const std::vector<Curve> &curves, std::size_t total_budget,
                                  AllocationStats *stats) {
    const std::size_t n = curves.size();
    if (total_budget < 2 * n)
        throw std::invalid_argument("allocate: budget below two endpoints per trajectory");
    for (const auto &c : curves)
        if (c.value.empty())
            throw std::invalid_argument("allocate: empty curve");

    std::vector<std::size_t> b(n, 2);
    std::size_t left = total_budget - 2 * n;

    // Greedy start
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

    auto sum = [&] {
        double s = 0;
        for (std::size_t i = 0; i < n; ++i)
            s += curves[i].at(b[i]);
        return s;
    };
    if (stats)
        stats->greedy_value = sum();

    // Exchange moves: k points from a donor to a receiver, the best move each time
    std::size_t exchanges = 0;
    for (; exchanges < kMaxExchanges; ++exchanges) {
        double best_delta = kMinGain;
        std::size_t best_k = 0, from = 0, to = 0;
        for (std::size_t k = 1; k <= kMaxStep; ++k) {
            Top2 gain, keep; // keep = minus the loss of giving k away
            for (std::size_t i = 0; i < n; ++i) {
                const auto &c = curves[i];
                if (b[i] + k <= c.max_budget())
                    gain.offer(c.at(b[i] + k) - c.at(b[i]), i);
                if (b[i] >= 2 + k)
                    keep.offer(c.at(b[i] - k) - c.at(b[i]), i);
            }
            // Best pair with donor != receiver
            const std::pair<double, std::size_t> receivers[] = {{gain.first, gain.first_i},
                                                                {gain.second, gain.second_i}};
            const std::pair<double, std::size_t> donors[] = {{keep.first, keep.first_i},
                                                             {keep.second, keep.second_i}};
            for (const auto &[g, r] : receivers)
                for (const auto &[l, d] : donors)
                    if (r != d && g + l > best_delta) {
                        best_delta = g + l;
                        best_k = k;
                        from = d;
                        to = r;
                    }
        }
        if (best_k == 0)
            break;
        b[from] -= best_k;
        b[to] += best_k;
    }

    if (stats) {
        stats->final_value = sum();
        stats->exchanges = exchanges;
    }
    return b;
}

} // namespace teacher

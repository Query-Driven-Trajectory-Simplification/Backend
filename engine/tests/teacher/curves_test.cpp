#include <gtest/gtest.h>

#include "backend/engine/teacher/curves.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

teacher::Curve curve(std::vector<double> values) {
    teacher::Curve c;
    c.method.assign(values.size(), 0);
    c.value = std::move(values);
    return c;
}

double value_of(const std::vector<teacher::Curve> &curves, const std::vector<std::size_t> &b) {
    double sum = 0;
    for (std::size_t i = 0; i < curves.size(); ++i)
        sum += curves[i].at(b[i]);
    return sum;
}

// Best total over every way to split the budget, for small cases
double brute_force(const std::vector<teacher::Curve> &curves, std::size_t budget) {
    double best = -INFINITY;
    std::vector<std::size_t> b(curves.size(), 2);
    auto recurse = [&](auto &&self, std::size_t i, std::size_t left) -> void {
        if (i + 1 == curves.size()) {
            if (left >= 2 && left <= curves[i].max_budget()) {
                b[i] = left;
                best = std::max(best, value_of(curves, b));
            }
            return;
        }
        for (std::size_t x = 2; x <= curves[i].max_budget() && x <= left; ++x) {
            b[i] = x;
            self(self, i + 1, left - x);
        }
    };
    recurse(recurse, 0, budget);
    return best;
}

} // namespace

TEST(TeacherAllocate, StepsOverFlatStretches) {
    // A only gains at its second extra point, B gains a little at once. Marginal greedy would
    // give the two points to B (0.35); gain per point over up to three points gives them to A
    const std::vector curves{curve({0.0, 0.0, 1.0}), curve({0.0, 0.3, 0.35})};
    const auto b = teacher::allocate(curves, 6);
    EXPECT_EQ(b, (std::vector<std::size_t>{4, 2}));
}

TEST(TeacherAllocate, KeepsTheExactBudgetAndBounds) {
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> step(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> length(1, 8);
    std::vector<teacher::Curve> curves;
    for (int i = 0; i < 50; ++i) {
        std::vector<double> v{0.0};
        for (std::size_t k = 1, n = length(rng); k < n; ++k)
            v.push_back(v.back() + step(rng) - 0.3); // not monotone
        curves.push_back(curve(v));
    }
    std::size_t capacity = 0;
    for (const auto &c : curves)
        capacity += c.max_budget();
    for (const std::size_t budget : {std::size_t{100}, std::size_t{180}, capacity}) {
        const auto b = teacher::allocate(curves, budget);
        EXPECT_EQ(std::accumulate(b.begin(), b.end(), std::size_t{0}), budget);
        for (std::size_t i = 0; i < curves.size(); ++i) {
            EXPECT_GE(b[i], 2u);
            EXPECT_LE(b[i], curves[i].max_budget());
        }
    }
}

TEST(TeacherAllocate, IsOptimalOnConcaveCurves) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> gain(0.0, 1.0);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<teacher::Curve> curves;
        for (int i = 0; i < 4; ++i) {
            std::vector<double> gains(5);
            for (auto &g : gains)
                g = gain(rng);
            std::ranges::sort(gains, std::greater<>());
            std::vector<double> v{0.0};
            for (const double g : gains)
                v.push_back(v.back() + g);
            curves.push_back(curve(v));
        }
        const std::size_t budget = 8 + static_cast<std::size_t>(trial % 12);
        const auto b = teacher::allocate(curves, budget);
        EXPECT_NEAR(value_of(curves, b), brute_force(curves, budget), 1e-9);
    }
}

TEST(TeacherAllocate, RejectsBudgetsThatDoNotFit) {
    const std::vector curves{curve({0.0, 1.0}), curve({0.0})};
    EXPECT_THROW(teacher::allocate(curves, 3), std::invalid_argument); // below 2 per trajectory
    EXPECT_THROW(teacher::allocate(curves, 6), std::runtime_error);    // more than 3 + 2 fits
    EXPECT_EQ(teacher::allocate(curves, 5), (std::vector<std::size_t>{3, 2}));
}

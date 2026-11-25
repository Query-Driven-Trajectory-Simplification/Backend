#include <gtest/gtest.h>

#include "backend/engine/teacher/objective.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>

namespace {

using teacher::flip_probability;

// The definition, integrated numerically: the flip chance at slack x, averaged over x uniform in
// [max(0, slack - bandwidth), slack + bandwidth]
double brute_force(double slack, double deviation, double bandwidth) {
    const double a = std::max(0.0, slack - bandwidth), b = slack + bandwidth;
    constexpr std::size_t steps = 200000;
    double sum = 0.0;
    for (std::size_t k = 0; k < steps; ++k) {
        const double x = a + (b - a) * (static_cast<double>(k) + 0.5) / steps;
        sum += std::max(0.0, deviation - x) / (2.0 * deviation);
    }
    return sum / steps;
}

TEST(FlipProbability, IsZeroWithoutDeviation) {
    for (const double slack : {0.0, 5.0, 100.0})
        for (const double bandwidth : {0.0, 10.0, 50.0})
            EXPECT_EQ(flip_probability(slack, 0.0, bandwidth), 0.0);
}

TEST(FlipProbability, WithoutBandwidthIsTheKnnMarginRule) {
    // The margin term's survival: 1 up to the gap, then (1 + gap / deviation) / 2
    for (const double slack : {0.0, 10.0, 30.0})
        for (const double deviation : {1.0, 10.0, 29.0, 30.0, 45.0, 300.0}) {
            const double survival = deviation <= slack ? 1.0 : 0.5 * (1.0 + slack / deviation);
            EXPECT_NEAR(flip_probability(slack, deviation, 0.0), 1.0 - survival, 1e-15);
        }
}

TEST(FlipProbability, MatchesTheAverageItStandsFor) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> slack(0.0, 100.0), deviation(0.1, 200.0),
        bandwidth(0.5, 60.0);
    for (int trial = 0; trial < 200; ++trial) {
        const double s = slack(rng), d = deviation(rng), h = bandwidth(rng);
        EXPECT_NEAR(flip_probability(s, d, h), brute_force(s, d, h), 1e-7)
            << "slack " << s << ", deviation " << d << ", bandwidth " << h;
    }
}

TEST(FlipProbability, WorkedExample) {
    // Slack 30 m, bandwidth 15 m: slacks 15..45 m. Deviation 29 m, just under the pair's own
    // slack, is free without a bandwidth but not with one
    EXPECT_EQ(flip_probability(30.0, 29.0, 0.0), 0.0);
    // (G(29) - G(15)) / (2 * 29 * 30) with G(x) = 29 x - x^2 / 2
    EXPECT_NEAR(flip_probability(30.0, 29.0, 15.0), 98.0 / 1740.0, 1e-15);
    // Only slacks below the deviation can flip: none here
    EXPECT_EQ(flip_probability(30.0, 15.0, 15.0), 0.0);
}

TEST(FlipProbability, StaysWithinAHalfAndReachesItForAnInfiniteDeviation) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    for (const double slack : {0.0, 20.0, 150.0})
        for (const double bandwidth : {0.0, 10.0, 80.0}) {
            EXPECT_EQ(flip_probability(slack, inf, bandwidth), 0.5);
            for (const double deviation : {1.0, 50.0, 1e3, 1e6}) {
                const double p = flip_probability(slack, deviation, bandwidth);
                EXPECT_GE(p, 0.0);
                EXPECT_LE(p, 0.5);
            }
            EXPECT_NEAR(flip_probability(slack, 1e9, bandwidth), 0.5, 1e-6);
        }
    // A pair at the threshold flips half the time without a bandwidth, whatever the deviation
    EXPECT_EQ(flip_probability(0.0, 10.0, 0.0), 0.5);
}

TEST(FlipProbability, RisesWithDeviationAndFallsWithSlack) {
    for (const double bandwidth : {0.0, 5.0, 25.0}) {
        for (const double slack : {0.0, 10.0, 40.0}) {
            double previous = 0.0;
            for (double deviation = 0.5; deviation < 200.0; deviation += 0.5) {
                const double p = flip_probability(slack, deviation, bandwidth);
                EXPECT_GE(p, previous - 1e-15) << "slack " << slack << ", deviation " << deviation;
                previous = p;
            }
        }
        for (const double deviation : {5.0, 30.0, 120.0}) {
            double previous = 0.5;
            for (double slack = 0.0; slack < 200.0; slack += 0.5) {
                const double p = flip_probability(slack, deviation, bandwidth);
                EXPECT_LE(p, previous + 1e-15) << "slack " << slack << ", deviation " << deviation;
                previous = p;
            }
        }
    }
}

TEST(FlipProbability, ABandwidthChargesDeviationsBelowTheSlack) {
    // Without a bandwidth nothing below the slack costs anything; with one, the charge starts at
    // slack - bandwidth
    EXPECT_EQ(flip_probability(30.0, 20.0, 0.0), 0.0);
    EXPECT_EQ(flip_probability(30.0, 20.0, 10.0), 0.0);
    EXPECT_GT(flip_probability(30.0, 20.0 + 1e-6, 10.0), 0.0);
    EXPECT_GT(flip_probability(30.0, 20.0, 15.0), 0.0);
}

} // namespace

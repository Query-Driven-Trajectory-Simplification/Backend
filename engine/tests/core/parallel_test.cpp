#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "backend/engine/core/parallel.hpp"

TEST(ParallelFor, ZeroIterationsNeverCallsFn) {
    bool called = false;
    core::parallel_for(0, [&](std::size_t) { called = true; });
    EXPECT_FALSE(called);
}

TEST(ParallelFor, CallsEveryIndexExactlyOnce) {
    constexpr std::size_t n = 10000;
    std::vector<std::atomic<int>> calls(n);
    core::parallel_for(n, [&](std::size_t i) { calls[i]++; });
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(calls[i], 1) << "index " << i;
    }
}

TEST(ParallelFor, FewerIndicesThanThreads) {
    std::vector<std::atomic<int>> calls(2);
    core::parallel_for(2, [&](std::size_t i) { calls[i]++; });
    EXPECT_EQ(calls[0], 1);
    EXPECT_EQ(calls[1], 1);
}

TEST(ParallelFor, RethrowsExceptionFromFn) {
    EXPECT_THROW(core::parallel_for(1000,
                                    [](std::size_t i) {
                                        if (i == 500) {
                                            throw std::runtime_error("boom");
                                        }
                                    }),
                 std::runtime_error);
}

TEST(ParallelFor, StopsClaimingIndicesAfterException) {
    constexpr std::size_t n = 1000000;
    std::atomic<std::size_t> calls{0};
    EXPECT_THROW(core::parallel_for(n,
                                    [&](std::size_t) {
                                        calls++;
                                        throw std::runtime_error("boom");
                                    }),
                 std::runtime_error);
    EXPECT_LT(calls.load(), n);
}

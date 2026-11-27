#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/range_query.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace student {

struct SampledRangeBox {
    query::RangeBox box;
    double weight = 0.0; // exact 1.0 / |reference| recall weight
};

// Fast 8-cell spatial-temporal grid sampler generating synthetic valid [5, 20] Range boxes
// directly from db (zero Q leakage), pre-filtered to boxes NOT already covered by endpoints.
struct SampledRangeIndex {
    std::vector<std::vector<SampledRangeBox>> uncov_boxes; // per-trajectory uncovered boxes
    std::vector<double> uncov_weight;                      // scale-invariant sum(w_b) * N / valid_boxes
    std::vector<double> ep_weight;                         // scale-invariant endpoint-covered mass
    std::vector<double> uncov_count_norm;                  // scale-invariant box count * N / valid_boxes
    std::size_t valid_boxes = 0;
};

SampledRangeIndex build_sampled_range_index(const core::TrajectoryDb &db,
                                            std::size_t target_boxes = 128000,
                                            std::uint64_t seed = 0x9E3779B97F4A7C15ULL);

} // namespace student

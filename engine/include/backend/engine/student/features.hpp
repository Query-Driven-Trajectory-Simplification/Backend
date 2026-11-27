#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/student/density_index.hpp"
#include "backend/engine/student/range_sampler.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace student {

struct TrajectoryFeatureTable {
    std::vector<std::string> feature_names;
    std::size_t num_trajs = 0;
    // Row-major [num_trajs x feature_names.size()]
    std::vector<float> values;
};

// Extract the compact 13-feature zero-Frechet set:
//   Single-trip (6): log_n, log_arc, log_cells_1km, log_rdp_err_2, log_rdp_err_8, log_sed_err_8
//   Database    (7): rho_mean, log_rho_sum, sig5_cnt_200, log_sig5_d1,
//                    log_rng_uncov, log_rng_boxes, rng_ep_frac
TrajectoryFeatureTable extract_trajectory_features(const core::TrajectoryDb &db,
                                                   const DbDensityIndex &density_idx,
                                                   const SampledRangeIndex *range_idx = nullptr);

} // namespace student

#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace student {

struct CandidatePoolConfig {
    double sed_tol_m = 5.0;
    double arc_stride_m = 250.0;
    double cell_m = 1000.0;
};

// Cumulative arc length in meters along a trajectory.
std::vector<double> compute_arc_length(const core::TrajectoryView &traj);

// union of top-down SED (> sed_tol_m), arc-length stride (<= arc_stride_m),
// and spatial cell-run midpoints (cell_m). Sorted, endpoints always included.
std::vector<std::size_t> build_candidate_pool(const core::TrajectoryView &traj,
                                              const CandidatePoolConfig &cfg);

// Lightweight O(N) 3D spatiotemporal density hash built strictly from raw DB (zero Frechet).
class DbDensityIndex {
  public:
    struct CellStats {
        std::uint32_t point_count = 0;
        std::uint32_t traj_count = 0;
    };

    DbDensityIndex(const core::TrajectoryDb &db, double cell_m = 333.3333,
                   std::int64_t time_bin_s = 302400);

    // Normalized per-point Range importance proxy along a trajectory.
    std::vector<float> trajectory_range_rho(const core::TrajectoryView &traj) const;

    // Query raw 3D cell counts at grid coordinates (gx, gy, gt).
    CellStats cell_stats(std::int32_t gx, std::int32_t gy, std::int32_t gt) const;

    double cell_m() const { return m_cell_m; }
    std::int64_t time_bin_s() const { return m_time_bin_s; }

  private:
    static std::uint64_t pack_3d(std::int32_t gx, std::int32_t gy, std::int32_t gt);

    double m_cell_m;
    std::int64_t m_time_bin_s;
    double m_mean_raw_imp = 1.0;
    std::unordered_map<std::uint64_t, CellStats> m_st_cells;
};

} // namespace student

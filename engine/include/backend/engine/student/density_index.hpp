#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace student {

// Lightweight O(N) 3D spatiotemporal density hash built strictly from raw DB (zero Frechet).
class DbDensityIndex {
  public:
    struct CellStats {
        std::uint32_t point_count = 0;
        std::uint32_t traj_count = 0;
    };

    explicit DbDensityIndex(const core::TrajectoryDb &db, double cell_m = 333.3333,
                            std::int64_t time_bin_s = 302400);

    // Normalized per-point Range importance proxy along a trajectory.
    std::vector<float> trajectory_range_rho(const core::TrajectoryView &traj) const;

    // Query raw 3D cell counts at grid coordinates (gx, gy, gt).
    CellStats cell_stats(std::int32_t gx, std::int32_t gy, std::int32_t gt) const;

    double cell_m() const { return m_cell_m; }
    std::int64_t time_bin_s() const { return m_time_bin_s; }

    static std::uint64_t pack_3d(std::int32_t gx, std::int32_t gy, std::int32_t gt);

  private:
    double m_cell_m;
    std::int64_t m_time_bin_s;
    double m_mean_raw_imp = 1.0;
    std::unordered_map<std::uint64_t, CellStats> m_st_cells;
};

} // namespace student

#include "backend/engine/student/density_index.hpp"

#include "backend/engine/core/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace student {

std::uint64_t DbDensityIndex::pack_3d(std::int32_t gx, std::int32_t gy, std::int32_t gt) {
    const std::uint64_t ux = static_cast<std::uint64_t>(static_cast<std::int64_t>(gx) + (1 << 20)) &
                             ((1ULL << 21) - 1);
    const std::uint64_t uy = static_cast<std::uint64_t>(static_cast<std::int64_t>(gy) + (1 << 20)) &
                             ((1ULL << 21) - 1);
    const std::uint64_t ut = static_cast<std::uint64_t>(static_cast<std::int64_t>(gt) + (1 << 20)) &
                             ((1ULL << 21) - 1);
    return (ux << 42) | (uy << 21) | ut;
}

DbDensityIndex::DbDensityIndex(const core::TrajectoryDb &db, double cell_m,
                               std::int64_t time_bin_s)
    : m_cell_m(cell_m), m_time_bin_s(time_bin_s) {
    const std::size_t num_trajs = db.size();
    const double inv_cell = 1.0 / m_cell_m;
    const double inv_time = 1.0 / static_cast<double>(m_time_bin_s);

    const std::size_t num_chunks =
        std::min<std::size_t>(16, std::max<std::size_t>(1, (num_trajs + 255) / 256));
    std::vector<std::unordered_map<std::uint64_t, CellStats>> chunk_cells(num_chunks);

    core::parallel_for(num_chunks, [&](std::size_t chunk_idx) {
        const std::size_t start_traj = (chunk_idx * num_trajs) / num_chunks;
        const std::size_t end_traj = ((chunk_idx + 1) * num_trajs) / num_chunks;
        auto &local_cells = chunk_cells[chunk_idx];
        local_cells.reserve((end_traj - start_traj) * 12);
        std::vector<std::uint64_t> visited_cells;
        visited_cells.reserve(256);

        for (std::size_t traj_idx = start_traj; traj_idx < end_traj; ++traj_idx) {
            const auto traj = db.trajectory(traj_idx);
            visited_cells.clear();
            std::uint64_t prev_key = 0;
            std::uint32_t run_count = 0;
            for (std::size_t pt_idx = 0; pt_idx < traj.size(); ++pt_idx) {
                const auto gx = static_cast<std::int32_t>(std::floor(traj.x[pt_idx] * inv_cell));
                const auto gy = static_cast<std::int32_t>(std::floor(traj.y[pt_idx] * inv_cell));
                const auto gt = static_cast<std::int32_t>(
                    std::floor(static_cast<double>(traj.t[pt_idx]) * inv_time));
                const std::uint64_t key = pack_3d(gx, gy, gt);
                if (run_count > 0 && key == prev_key) {
                    ++run_count;
                } else {
                    if (run_count > 0) {
                        local_cells[prev_key].point_count += run_count;
                        visited_cells.push_back(prev_key);
                    }
                    prev_key = key;
                    run_count = 1;
                }
            }
            if (run_count > 0) {
                local_cells[prev_key].point_count += run_count;
                visited_cells.push_back(prev_key);
            }
            std::ranges::sort(visited_cells);
            visited_cells.erase(std::unique(visited_cells.begin(), visited_cells.end()),
                                visited_cells.end());
            for (std::uint64_t key : visited_cells)
                local_cells[key].traj_count += 1;
        }
    });

    m_st_cells = std::move(chunk_cells[0]);
    m_st_cells.reserve(num_trajs * 12);
    for (std::size_t chunk_idx = 1; chunk_idx < num_chunks; ++chunk_idx) {
        for (const auto &[key, stats] : chunk_cells[chunk_idx]) {
            auto &merged = m_st_cells[key];
            merged.point_count += stats.point_count;
            merged.traj_count += stats.traj_count;
        }
    }

    double importance_sum = 0.0;
    std::size_t populated_cell_count = 0;
    for (const auto &[key, stats] : m_st_cells) {
        if (stats.traj_count >= 2) {
            const double traj_count = static_cast<double>(stats.traj_count);
            const double accept_prob = std::min(1.0, traj_count / 20.0);
            importance_sum += std::log1p(static_cast<double>(stats.point_count) * accept_prob /
                                         std::max(1.0, traj_count));
            ++populated_cell_count;
        }
    }
    m_mean_raw_imp = populated_cell_count > 0
                         ? std::max(1e-6, importance_sum / static_cast<double>(populated_cell_count))
                         : 1.0;
}

DbDensityIndex::CellStats DbDensityIndex::cell_stats(std::int32_t gx, std::int32_t gy,
                                                     std::int32_t gt) const {
    const auto it = m_st_cells.find(pack_3d(gx, gy, gt));
    return it != m_st_cells.end() ? it->second : CellStats{};
}

std::vector<float> DbDensityIndex::trajectory_range_rho(const core::TrajectoryView &traj) const {
    const std::size_t num_pts = traj.size();
    std::vector<float> rho(num_pts, 0.0f);
    const double inv_cell = 1.0 / m_cell_m;
    const double inv_time = 1.0 / static_cast<double>(m_time_bin_s);

    bool has_prev_cell = false;
    std::uint64_t prev_key = 0;
    float prev_rho = 0.0f;
    for (std::size_t pt_idx = 0; pt_idx < num_pts; ++pt_idx) {
        const auto gx = static_cast<std::int32_t>(std::floor(traj.x[pt_idx] * inv_cell));
        const auto gy = static_cast<std::int32_t>(std::floor(traj.y[pt_idx] * inv_cell));
        const auto gt = static_cast<std::int32_t>(
            std::floor(static_cast<double>(traj.t[pt_idx]) * inv_time));
        const std::uint64_t key = pack_3d(gx, gy, gt);
        if (has_prev_cell && key == prev_key) {
            rho[pt_idx] = prev_rho;
            continue;
        }

        double point_sum = 0.0;
        double traj_sum = 0.0;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dt = -1; dt <= 1; ++dt) {
                    const auto it = m_st_cells.find(pack_3d(gx + dx, gy + dy, gt + dt));
                    if (it != m_st_cells.end()) {
                        const double neighbor_weight = (dx == 0 && dy == 0 && dt == 0) ? 1.0 : 0.25;
                        point_sum += neighbor_weight * static_cast<double>(it->second.point_count);
                        traj_sum += neighbor_weight * static_cast<double>(it->second.traj_count);
                    }
                }
            }
        }
        float cell_rho = 0.0f;
        if (traj_sum >= 3.0) {
            const double accept_prob = std::min(1.0, traj_sum / 20.0);
            const double importance =
                std::log1p(point_sum * accept_prob / std::max(1.0, traj_sum));
            cell_rho = static_cast<float>(importance / m_mean_raw_imp);
        }
        rho[pt_idx] = cell_rho;
        prev_key = key;
        prev_rho = cell_rho;
        has_prev_cell = true;
    }
    return rho;
}

} // namespace student

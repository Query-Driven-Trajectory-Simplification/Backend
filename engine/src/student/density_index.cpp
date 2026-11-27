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
    const std::size_t N = db.size();
    const double inv_cell = 1.0 / m_cell_m;
    const double inv_time = 1.0 / static_cast<double>(m_time_bin_s);

    const std::size_t num_chunks = std::min<std::size_t>(16, std::max<std::size_t>(1, (N + 255) / 256));
    std::vector<std::unordered_map<std::uint64_t, CellStats>> locals(num_chunks);

    core::parallel_for(num_chunks, [&](std::size_t c) {
        const std::size_t i0 = (c * N) / num_chunks;
        const std::size_t i1 = ((c + 1) * N) / num_chunks;
        auto &cells = locals[c];
        cells.reserve((i1 - i0) * 12);
        std::vector<std::uint64_t> seen;
        seen.reserve(256);

        for (std::size_t i = i0; i < i1; ++i) {
            const auto traj = db.trajectory(i);
            seen.clear();
            std::uint64_t prev_key = 0;
            std::uint32_t run = 0;
            for (std::size_t k = 0; k < traj.size(); ++k) {
                const auto gx = static_cast<std::int32_t>(std::floor(traj.x[k] * inv_cell));
                const auto gy = static_cast<std::int32_t>(std::floor(traj.y[k] * inv_cell));
                const auto gt = static_cast<std::int32_t>(
                    std::floor(static_cast<double>(traj.t[k]) * inv_time));
                const std::uint64_t key = pack_3d(gx, gy, gt);
                if (run > 0 && key == prev_key) {
                    ++run;
                } else {
                    if (run > 0) {
                        cells[prev_key].point_count += run;
                        seen.push_back(prev_key);
                    }
                    prev_key = key;
                    run = 1;
                }
            }
            if (run > 0) {
                cells[prev_key].point_count += run;
                seen.push_back(prev_key);
            }
            std::ranges::sort(seen);
            seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
            for (std::uint64_t key : seen)
                cells[key].traj_count += 1;
        }
    });

    m_st_cells = std::move(locals[0]);
    m_st_cells.reserve(N * 12);
    for (std::size_t c = 1; c < num_chunks; ++c) {
        for (const auto &[key, st] : locals[c]) {
            auto &dst = m_st_cells[key];
            dst.point_count += st.point_count;
            dst.traj_count += st.traj_count;
        }
    }

    double sum_imp = 0.0;
    std::size_t cnt_imp = 0;
    for (const auto &[key, st] : m_st_cells) {
        if (st.traj_count >= 2) {
            const double r = static_cast<double>(st.traj_count);
            const double acc = std::min(1.0, r / 20.0);
            sum_imp += std::log1p(static_cast<double>(st.point_count) * acc / std::max(1.0, r));
            ++cnt_imp;
        }
    }
    m_mean_raw_imp = cnt_imp > 0 ? std::max(1e-6, sum_imp / static_cast<double>(cnt_imp)) : 1.0;
}

DbDensityIndex::CellStats DbDensityIndex::cell_stats(std::int32_t gx, std::int32_t gy,
                                                     std::int32_t gt) const {
    const auto it = m_st_cells.find(pack_3d(gx, gy, gt));
    return it != m_st_cells.end() ? it->second : CellStats{};
}

std::vector<float> DbDensityIndex::trajectory_range_rho(const core::TrajectoryView &traj) const {
    const std::size_t n = traj.size();
    std::vector<float> rho(n, 0.0f);
    const double inv_cell = 1.0 / m_cell_m;
    const double inv_t = 1.0 / static_cast<double>(m_time_bin_s);

    bool has_prev = false;
    std::uint64_t prev_key = 0;
    float prev_rho = 0.0f;
    for (std::size_t k = 0; k < n; ++k) {
        const auto gx = static_cast<std::int32_t>(std::floor(traj.x[k] * inv_cell));
        const auto gy = static_cast<std::int32_t>(std::floor(traj.y[k] * inv_cell));
        const auto gt = static_cast<std::int32_t>(std::floor(static_cast<double>(traj.t[k]) * inv_t));
        const std::uint64_t key = pack_3d(gx, gy, gt);
        if (has_prev && key == prev_key) {
            rho[k] = prev_rho;
            continue;
        }

        double pt_sum = 0.0, tr_sum = 0.0;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dt = -1; dt <= 1; ++dt) {
                    const auto it = m_st_cells.find(pack_3d(gx + dx, gy + dy, gt + dt));
                    if (it != m_st_cells.end()) {
                        const double w = (dx == 0 && dy == 0 && dt == 0) ? 1.0 : 0.25;
                        pt_sum += w * static_cast<double>(it->second.point_count);
                        tr_sum += w * static_cast<double>(it->second.traj_count);
                    }
                }
            }
        }
        float val = 0.0f;
        if (tr_sum >= 3.0) {
            const double acc = std::min(1.0, tr_sum / 20.0);
            const double imp = std::log1p(pt_sum * acc / std::max(1.0, tr_sum));
            val = static_cast<float>(imp / m_mean_raw_imp);
        }
        rho[k] = val;
        prev_key = key;
        prev_rho = val;
        has_prev = true;
    }
    return rho;
}

} // namespace student

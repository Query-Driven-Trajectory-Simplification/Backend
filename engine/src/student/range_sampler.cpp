#include "backend/engine/student/range_sampler.hpp"

#include "backend/engine/core/parallel.hpp"
#include "backend/engine/student/density_index.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

namespace student {

namespace {

inline bool pt_in_box(const query::RangeBox &b, std::int64_t t, double x, double y) {
    return t >= b.t_min && t <= b.t_max && x >= b.x_min && x <= b.x_max && y >= b.y_min &&
           y <= b.y_max;
}

} // namespace

SampledRangeIndex build_sampled_range_index(const core::TrajectoryDb &db, std::size_t target_boxes,
                                            std::uint64_t seed) {
    const std::size_t N = db.size();
    SampledRangeIndex idx;
    idx.uncov_boxes.resize(N);
    idx.uncov_weight.assign(N, 0.0);
    idx.ep_weight.assign(N, 0.0);
    idx.uncov_count_norm.assign(N, 0.0);
    if (N < 5 || target_boxes == 0)
        return idx;

    // 1. Build exact 1km x 1km x 7d inverted grid of trajectory IDs (~18 ms on 16.26M pts)
    constexpr double kCellM = 1000.0;
    constexpr double kInvCell = 1.0 / kCellM;
    constexpr std::int64_t kTimeBin = 604800; // 7 days
    constexpr double kInvTime = 1.0 / static_cast<double>(kTimeBin);

    std::vector<std::size_t> pt_offsets(N + 1, 0);
    for (std::size_t i = 0; i < N; ++i)
        pt_offsets[i + 1] = pt_offsets[i] + db.trajectory(i).size();
    const std::size_t total_pts = pt_offsets.back();
    if (total_pts == 0)
        return idx;

    std::vector<std::vector<std::uint64_t>> traj_keys(N);
    core::parallel_for(N, [&](std::size_t i) {
        const auto tr = db.trajectory(i);
        auto &keys = traj_keys[i];
        keys.reserve(32);
        std::uint64_t prev = 0;
        for (std::size_t k = 0; k < tr.size(); ++k) {
            const auto gx = static_cast<std::int32_t>(std::floor(tr.x[k] * kInvCell));
            const auto gy = static_cast<std::int32_t>(std::floor(tr.y[k] * kInvCell));
            const auto gt = static_cast<std::int32_t>(
                std::floor(static_cast<double>(tr.t[k]) * kInvTime));
            const std::uint64_t key = DbDensityIndex::pack_3d(gx, gy, gt);
            if (k == 0 || key != prev) {
                keys.push_back(key);
                prev = key;
            }
        }
        std::ranges::sort(keys);
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    });

    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> grid;
    grid.reserve(N * 6);
    for (std::size_t i = 0; i < N; ++i)
        for (std::uint64_t key : traj_keys[i])
            grid[key].push_back(static_cast<std::uint32_t>(i));

    // 2. Sample candidate centres in parallel using 2x2x2 = 8 exact grid cells
    const std::size_t n_cand = static_cast<std::size_t>(target_boxes * 2.2);
    std::vector<std::uint64_t> rand_pts(n_cand);
    std::vector<double> rand_u(n_cand);
    std::mt19937_64 rng(seed);
    for (std::size_t c = 0; c < n_cand; ++c) {
        rand_pts[c] = rng() % total_pts;
        rand_u[c] = static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);
    }

    struct ValidBox {
        query::RangeBox box{};
        double weight = 0.0;
        std::vector<std::uint32_t> uncov_trajs;
        std::vector<std::uint32_t> ep_trajs;
    };
    std::vector<ValidBox> out(n_cand);
    std::vector<std::uint8_t> kept_flag(n_cand, 0);

    core::parallel_for(n_cand, [&](std::size_t c) {
        const std::size_t g_pt = rand_pts[c];
        const std::size_t ti = static_cast<std::size_t>(
            std::upper_bound(pt_offsets.begin(), pt_offsets.end(), g_pt) - pt_offsets.begin() - 1);
        const auto ctr = db.trajectory(ti);
        const std::size_t pt = g_pt - pt_offsets[ti];
        const query::RangeBox bx{
            ctr.x[pt] - 500.0, ctr.x[pt] + 500.0, ctr.y[pt] - 500.0,
            ctr.y[pt] + 500.0, ctr.t[pt] - 302400, ctr.t[pt] + 302400,
        };
        const auto gx0 = static_cast<std::int32_t>(std::floor(bx.x_min * kInvCell));
        const auto gx1 = static_cast<std::int32_t>(std::floor(bx.x_max * kInvCell));
        const auto gy0 = static_cast<std::int32_t>(std::floor(bx.y_min * kInvCell));
        const auto gy1 = static_cast<std::int32_t>(std::floor(bx.y_max * kInvCell));
        const auto gt0 =
            static_cast<std::int32_t>(std::floor(static_cast<double>(bx.t_min) * kInvTime));
        const auto gt1 =
            static_cast<std::int32_t>(std::floor(static_cast<double>(bx.t_max) * kInvTime));

        std::vector<std::uint32_t> cands;
        cands.reserve(64);
        for (std::int32_t gx = gx0; gx <= gx1; ++gx)
            for (std::int32_t gy = gy0; gy <= gy1; ++gy)
                for (std::int32_t gt = gt0; gt <= gt1; ++gt) {
                    const auto it = grid.find(DbDensityIndex::pack_3d(gx, gy, gt));
                    if (it != grid.end())
                        cands.insert(cands.end(), it->second.begin(), it->second.end());
                }
        if (cands.size() < 5)
            return;
        std::ranges::sort(cands);
        cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
        if (cands.size() < 5)
            return;

        std::size_t ref_cnt = 0;
        std::vector<std::uint32_t> uncov, ep_cov;
        uncov.reserve(cands.size());
        for (std::uint32_t j : cands) {
            const auto tr = db.trajectory(j);
            if (!query::intersects(tr, bx))
                continue;
            ++ref_cnt;
            const bool ep0 = pt_in_box(bx, tr.t.front(), tr.x.front(), tr.y.front());
            const bool ep1 = pt_in_box(bx, tr.t.back(), tr.x.back(), tr.y.back());
            if (ep0 || ep1)
                ep_cov.push_back(j);
            else
                uncov.push_back(j);
        }
        if (ref_cnt < 5)
            return;
        if (ref_cnt < 20 && rand_u[c] * 20.0 >= static_cast<double>(ref_cnt))
            return;

        kept_flag[c] = 1;
        out[c].box = bx;
        out[c].weight = 1.0 / static_cast<double>(ref_cnt);
        out[c].uncov_trajs = std::move(uncov);
        out[c].ep_trajs = std::move(ep_cov);
    });

    // 3. Invert valid boxes into per-trajectory uncovered lists + exact scale-invariant stats
    for (std::size_t c = 0; c < n_cand && idx.valid_boxes < target_boxes; ++c) {
        if (!kept_flag[c])
            continue;
        ++idx.valid_boxes;
        const double w = out[c].weight;
        for (std::uint32_t j : out[c].uncov_trajs) {
            idx.uncov_boxes[j].push_back({out[c].box, w});
            idx.uncov_weight[j] += w;
        }
        for (std::uint32_t j : out[c].ep_trajs)
            idx.ep_weight[j] += w;
    }

    if (idx.valid_boxes > 0) {
        const double scale = static_cast<double>(N) / static_cast<double>(idx.valid_boxes);
        for (std::size_t i = 0; i < N; ++i) {
            idx.uncov_weight[i] *= scale;
            idx.ep_weight[i] *= scale;
            idx.uncov_count_norm[i] = static_cast<double>(idx.uncov_boxes[i].size()) * scale;
        }
    }
    return idx;
}

} // namespace student

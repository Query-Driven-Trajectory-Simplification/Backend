#include "backend/engine/student/range_sampler.hpp"

#include "backend/engine/core/parallel.hpp"
#include "backend/engine/student/density_index.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

namespace student {

namespace {

inline bool point_in_box(const query::RangeBox &box, std::int64_t t, double x, double y) {
    return t >= box.t_min && t <= box.t_max && x >= box.x_min && x <= box.x_max && y >= box.y_min &&
           y <= box.y_max;
}

} // namespace

SampledRangeIndex build_sampled_range_index(const core::TrajectoryDb &db, std::size_t target_boxes,
                                            std::uint64_t seed) {
    const std::size_t num_trajs = db.size();
    SampledRangeIndex index;
    index.uncov_boxes.resize(num_trajs);
    index.uncov_weight.assign(num_trajs, 0.0);
    index.ep_weight.assign(num_trajs, 0.0);
    index.uncov_count_norm.assign(num_trajs, 0.0);
    if (num_trajs < 5 || target_boxes == 0)
        return index;

    // 1. Build exact 1km x 1km x 7d inverted grid of trajectory IDs (~18 ms on 16.26M pts)
    constexpr double kCellMeters = 1000.0;
    constexpr double kInvCell = 1.0 / kCellMeters;
    constexpr std::int64_t kTimeBinSec = 604800; // 7 days
    constexpr double kInvTime = 1.0 / static_cast<double>(kTimeBinSec);

    std::vector<std::size_t> point_offsets(num_trajs + 1, 0);
    for (std::size_t traj_idx = 0; traj_idx < num_trajs; ++traj_idx)
        point_offsets[traj_idx + 1] = point_offsets[traj_idx] + db.trajectory(traj_idx).size();
    const std::size_t total_points = point_offsets.back();
    if (total_points == 0)
        return index;

    std::vector<std::vector<std::uint64_t>> traj_cell_keys(num_trajs);
    core::parallel_for(num_trajs, [&](std::size_t traj_idx) {
        const auto traj = db.trajectory(traj_idx);
        auto &keys = traj_cell_keys[traj_idx];
        keys.reserve(32);
        std::uint64_t prev_key = 0;
        for (std::size_t pt_idx = 0; pt_idx < traj.size(); ++pt_idx) {
            const auto gx = static_cast<std::int32_t>(std::floor(traj.x[pt_idx] * kInvCell));
            const auto gy = static_cast<std::int32_t>(std::floor(traj.y[pt_idx] * kInvCell));
            const auto gt = static_cast<std::int32_t>(
                std::floor(static_cast<double>(traj.t[pt_idx]) * kInvTime));
            const std::uint64_t key = DbDensityIndex::pack_3d(gx, gy, gt);
            if (pt_idx == 0 || key != prev_key) {
                keys.push_back(key);
                prev_key = key;
            }
        }
        std::ranges::sort(keys);
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    });

    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> inverted_grid;
    inverted_grid.reserve(num_trajs * 6);
    for (std::size_t traj_idx = 0; traj_idx < num_trajs; ++traj_idx)
        for (std::uint64_t key : traj_cell_keys[traj_idx])
            inverted_grid[key].push_back(static_cast<std::uint32_t>(traj_idx));

    // 2. Sample candidate centres in parallel using 2x2x2 = 8 exact grid cell lookups
    const std::size_t num_candidates = static_cast<std::size_t>(target_boxes * 2.2);
    std::vector<std::uint64_t> random_point_ids(num_candidates);
    std::vector<double> accept_draws(num_candidates);
    std::mt19937_64 rng(seed);
    for (std::size_t cand_idx = 0; cand_idx < num_candidates; ++cand_idx) {
        random_point_ids[cand_idx] = rng() % total_points;
        accept_draws[cand_idx] = static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);
    }

    struct ValidBox {
        query::RangeBox box{};
        double weight = 0.0;
        std::vector<std::uint32_t> uncovered_trajs;
        std::vector<std::uint32_t> endpoint_covered_trajs;
    };
    std::vector<ValidBox> sampled(num_candidates);
    std::vector<std::uint8_t> is_accepted(num_candidates, 0);

    core::parallel_for(num_candidates, [&](std::size_t cand_idx) {
        const std::size_t global_pt = random_point_ids[cand_idx];
        const std::size_t center_traj_idx = static_cast<std::size_t>(
            std::upper_bound(point_offsets.begin(), point_offsets.end(), global_pt) -
            point_offsets.begin() - 1);
        const auto center_traj = db.trajectory(center_traj_idx);
        const std::size_t local_pt = global_pt - point_offsets[center_traj_idx];
        const query::RangeBox box{
            center_traj.x[local_pt] - 500.0, center_traj.x[local_pt] + 500.0,
            center_traj.y[local_pt] - 500.0, center_traj.y[local_pt] + 500.0,
            center_traj.t[local_pt] - 302400, center_traj.t[local_pt] + 302400,
        };
        const auto gx_min = static_cast<std::int32_t>(std::floor(box.x_min * kInvCell));
        const auto gx_max = static_cast<std::int32_t>(std::floor(box.x_max * kInvCell));
        const auto gy_min = static_cast<std::int32_t>(std::floor(box.y_min * kInvCell));
        const auto gy_max = static_cast<std::int32_t>(std::floor(box.y_max * kInvCell));
        const auto gt_min =
            static_cast<std::int32_t>(std::floor(static_cast<double>(box.t_min) * kInvTime));
        const auto gt_max =
            static_cast<std::int32_t>(std::floor(static_cast<double>(box.t_max) * kInvTime));

        std::vector<std::uint32_t> candidate_trajs;
        candidate_trajs.reserve(64);
        for (std::int32_t gx = gx_min; gx <= gx_max; ++gx)
            for (std::int32_t gy = gy_min; gy <= gy_max; ++gy)
                for (std::int32_t gt = gt_min; gt <= gt_max; ++gt) {
                    const auto it = inverted_grid.find(DbDensityIndex::pack_3d(gx, gy, gt));
                    if (it != inverted_grid.end())
                        candidate_trajs.insert(candidate_trajs.end(), it->second.begin(),
                                               it->second.end());
                }
        if (candidate_trajs.size() < 5)
            return;
        std::ranges::sort(candidate_trajs);
        candidate_trajs.erase(std::unique(candidate_trajs.begin(), candidate_trajs.end()),
                              candidate_trajs.end());
        if (candidate_trajs.size() < 5)
            return;

        std::size_t match_count = 0;
        std::vector<std::uint32_t> uncovered_trajs;
        std::vector<std::uint32_t> endpoint_covered_trajs;
        uncovered_trajs.reserve(candidate_trajs.size());
        for (std::uint32_t traj_idx : candidate_trajs) {
            const auto traj = db.trajectory(traj_idx);
            if (!query::intersects(traj, box))
                continue;
            ++match_count;
            const bool start_inside =
                point_in_box(box, traj.t.front(), traj.x.front(), traj.y.front());
            const bool end_inside = point_in_box(box, traj.t.back(), traj.x.back(), traj.y.back());
            if (start_inside || end_inside)
                endpoint_covered_trajs.push_back(traj_idx);
            else
                uncovered_trajs.push_back(traj_idx);
        }
        if (match_count < 5)
            return;
        if (match_count < 20 && accept_draws[cand_idx] * 20.0 >= static_cast<double>(match_count))
            return;

        is_accepted[cand_idx] = 1;
        sampled[cand_idx].box = box;
        sampled[cand_idx].weight = 1.0 / static_cast<double>(match_count);
        sampled[cand_idx].uncovered_trajs = std::move(uncovered_trajs);
        sampled[cand_idx].endpoint_covered_trajs = std::move(endpoint_covered_trajs);
    });

    // 3. Invert valid boxes into per-trajectory uncovered lists + exact scale-invariant stats
    for (std::size_t cand_idx = 0; cand_idx < num_candidates && index.valid_boxes < target_boxes;
         ++cand_idx) {
        if (!is_accepted[cand_idx])
            continue;
        ++index.valid_boxes;
        const double box_weight = sampled[cand_idx].weight;
        for (std::uint32_t traj_idx : sampled[cand_idx].uncovered_trajs) {
            index.uncov_boxes[traj_idx].push_back({sampled[cand_idx].box, box_weight});
            index.uncov_weight[traj_idx] += box_weight;
        }
        for (std::uint32_t traj_idx : sampled[cand_idx].endpoint_covered_trajs)
            index.ep_weight[traj_idx] += box_weight;
    }

    if (index.valid_boxes > 0) {
        const double norm_scale =
            static_cast<double>(num_trajs) / static_cast<double>(index.valid_boxes);
        for (std::size_t traj_idx = 0; traj_idx < num_trajs; ++traj_idx) {
            index.uncov_weight[traj_idx] *= norm_scale;
            index.ep_weight[traj_idx] *= norm_scale;
            index.uncov_count_norm[traj_idx] =
                static_cast<double>(index.uncov_boxes[traj_idx].size()) * norm_scale;
        }
    }
    return index;
}

} // namespace student

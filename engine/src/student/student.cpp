#include "backend/engine/student/student.hpp"

#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/timer.hpp"
#include "backend/engine/selection/rdp.hpp"
#include "backend/engine/student/candidate_pool.hpp"
#include "backend/engine/student/density_index.hpp"
#include "backend/engine/student/features.hpp"
#include "backend/engine/student/range_sampler.hpp"
#include "backend/engine/student/submodular_solver.hpp"

#include <chrono>
#include <stdexcept>
#include <vector>

namespace student {

SimplifyResult simplify_database_oof(const core::TrajectoryDb &db, std::size_t total_budget,
                                     std::span<const StudentModel> fold_models,
                                     std::span<const std::uint32_t> fold_ids, int rb_passes,
                                     bool rdp_only) {
    if (fold_models.empty())
        throw std::invalid_argument("fold_models cannot be empty");
    const std::size_t N = db.size();

    SimplifyResult res;

    // 1. 3D spatiotemporal density index + fast 128k sampled Range boxes (zero Frechet, zero Q leakage)
    auto t0 = std::chrono::steady_clock::now();
    const DbDensityIndex density_idx(db);
    const auto range_idx = build_sampled_range_index(db, 128000);
    res.density_ms = core::seconds_since(t0) * 1000.0;

    // 2. 13-feature extraction (tiny10 + 3 exact sampled Range features)
    auto t1 = std::chrono::steady_clock::now();
    const auto feats = extract_trajectory_features(db, density_idx, &range_idx);
    res.feature_ms = core::seconds_since(t1) * 1000.0;

    // 3. Stage 1 MLP -> exact discrete water-filling -> Stage 2 MLP
    auto t2 = std::chrono::steady_clock::now();
    std::vector<std::size_t> lengths(N);
    for (std::size_t i = 0; i < N; ++i)
        lengths[i] = db.trajectory(i).size();
    const auto ell = predict_marginal_curves(fold_models, fold_ids, feats, lengths);
    const auto budgets =
        discrete_water_filling(ell, lengths, total_budget, fold_models[0].log_anchors);
    std::vector<SubmodularWeights> weights;
    if (!rdp_only)
        weights =
            predict_submodular_weights(fold_models, fold_ids, feats, lengths, budgets, rb_passes);
    res.ml_ms = core::seconds_since(t2) * 1000.0;

    res.kept.resize(N);
    if (rdp_only) {
        auto t4 = std::chrono::steady_clock::now();
        core::parallel_for(N, [&](std::size_t i) {
            res.kept[i] = selection::rdp::simplify(db.trajectory(i), budgets[i]);
        });
        res.decode_ms = core::seconds_since(t4) * 1000.0;
        return res;
    }

    // 4. Safe Candidate Pool (~4x reduction)
    auto t3 = std::chrono::steady_clock::now();
    std::vector<std::vector<std::size_t>> pools(N);
    core::parallel_for(N, [&](std::size_t i) {
        pools[i] = build_candidate_pool(db.trajectory(i));
    });
    res.pool_ms = core::seconds_since(t3) * 1000.0;

    // 5. C++ Submodular Greedy + Red-Black Free-Knot Polish
    auto t4 = std::chrono::steady_clock::now();
    const bool use_boxes = range_idx.valid_boxes > 0;
    core::parallel_for(N, [&](std::size_t i) {
        const double tot_w = range_idx.uncov_weight[i] + range_idx.ep_weight[i];
        const double ufrac = (tot_w > 1e-12) ? (range_idx.uncov_weight[i] / tot_w) : 0.0;
        res.kept[i] = submodular_simplify_on_pool(
            db.trajectory(i), pools[i], budgets[i], weights[i], density_idx,
            use_boxes ? &range_idx.uncov_boxes[i] : nullptr, ufrac);
    });
    res.decode_ms = core::seconds_since(t4) * 1000.0;
    return res;
}

SimplifyResult simplify_database(const core::TrajectoryDb &db, std::size_t total_budget,
                                 const StudentModel &model, int rb_passes, bool rdp_only) {
    return simplify_database_oof(db, total_budget, std::span<const StudentModel>(&model, 1), {},
                                 rb_passes, rdp_only);
}

} // namespace student

#include "backend/engine/student/student.hpp"

#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/timer.hpp"
#include "backend/engine/selection/rdp.hpp"
#include "backend/engine/student/candidate_pool.hpp"
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
    const std::size_t num_trajs = db.size();

    SimplifyResult result;

    // 1. Fast 128k sampled Range boxes directly from DB (zero Frechet, zero Q leakage)
    auto t_range = std::chrono::steady_clock::now();
    const auto range_idx = build_sampled_range_index(db, 128000);
    result.range_index_ms = core::seconds_since(t_range) * 1000.0;

    // 2. 11-feature extraction (6 single-trip + 5 database context)
    auto t_features = std::chrono::steady_clock::now();
    const auto features = extract_trajectory_features(db, &range_idx);
    result.feature_ms = core::seconds_since(t_features) * 1000.0;

    // 3. Stage 1 MLP -> exact discrete water-filling -> Stage 2 MLP
    auto t_ml = std::chrono::steady_clock::now();
    std::vector<std::size_t> lengths(num_trajs);
    for (std::size_t i = 0; i < num_trajs; ++i)
        lengths[i] = db.trajectory(i).size();
    const auto marginal_curves = predict_marginal_curves(fold_models, fold_ids, features, lengths);
    const auto budgets = discrete_water_filling(marginal_curves, lengths, total_budget,
                                                fold_models[0].log_anchors);
    std::vector<SubmodularWeights> weights;
    if (!rdp_only)
        weights = predict_submodular_weights(fold_models, fold_ids, features, lengths, budgets,
                                             rb_passes);
    result.ml_ms = core::seconds_since(t_ml) * 1000.0;

    result.kept.resize(num_trajs);
    if (rdp_only) {
        auto t_decode = std::chrono::steady_clock::now();
        core::parallel_for(num_trajs, [&](std::size_t i) {
            result.kept[i] = selection::rdp::simplify(db.trajectory(i), budgets[i]);
        });
        result.decode_ms = core::seconds_since(t_decode) * 1000.0;
        return result;
    }

    // 4. Safe Candidate Pool (~4x reduction)
    auto t_pool = std::chrono::steady_clock::now();
    std::vector<std::vector<std::size_t>> pools(num_trajs);
    core::parallel_for(num_trajs, [&](std::size_t i) {
        pools[i] = build_candidate_pool(db.trajectory(i));
    });
    result.pool_ms = core::seconds_since(t_pool) * 1000.0;

    // 5. C++ Submodular Greedy + Red-Black Free-Knot Polish
    auto t_decode = std::chrono::steady_clock::now();
    core::parallel_for(num_trajs, [&](std::size_t i) {
        const double total_range_weight = range_idx.uncov_weight[i] + range_idx.ep_weight[i];
        const double uncovered_frac =
            (total_range_weight > 1e-12) ? (range_idx.uncov_weight[i] / total_range_weight) : 0.0;
        result.kept[i] = submodular_simplify_on_pool(db.trajectory(i), pools[i], budgets[i],
                                                     weights[i], range_idx.uncov_boxes[i],
                                                     uncovered_frac);
    });
    result.decode_ms = core::seconds_since(t_decode) * 1000.0;
    return result;
}

SimplifyResult simplify_database(const core::TrajectoryDb &db, std::size_t total_budget,
                                 const StudentModel &model, int rb_passes, bool rdp_only) {
    return simplify_database_oof(db, total_budget, std::span<const StudentModel>(&model, 1), {},
                                 rb_passes, rdp_only);
}

} // namespace student

#pragma once

#include "backend/engine/student/features.hpp"
#include "backend/engine/student/submodular_solver.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace student {

struct MlpLayer {
    std::uint32_t in_dim = 0;
    std::uint32_t out_dim = 0;
    bool has_layernorm = false; // true => Linear -> LayerNorm -> SiLU; false => Linear only
    std::vector<float> weight;  // [out_dim x in_dim], row-major
    std::vector<float> bias;    // [out_dim]
    std::vector<float> ln_gamma;
    std::vector<float> ln_beta;
};

struct StudentModel {
    std::uint32_t s1_in_dim = 0;
    std::uint32_t num_anchors = 0;
    std::vector<float> s1_mean, s1_std, log_anchors;
    std::vector<MlpLayer> s1_layers;

    std::uint32_t s2_in_dim = 0;
    std::uint32_t num_presets = 0;
    std::vector<float> s2_mean, s2_std, log_simplex; // log_simplex: [num_presets x 3]
    std::vector<MlpLayer> s2_layers;
};

// Load a single-file binary Student model exported by student/train_student.py.
StudentModel load_model(const std::filesystem::path &path);

// Stage 1 MLP: predicts row-major [N x num_anchors] decreasing log-marginal curves.
// When multiple fold_models are passed, each trajectory i uses fold_models[fold_ids[i]]
// and each fold's curves are dual-price aligned at 1% reference share.
std::vector<double> predict_marginal_curves(std::span<const StudentModel> fold_models,
                                            std::span<const std::uint32_t> fold_ids,
                                            const TrajectoryFeatureTable &feats,
                                            std::span<const std::size_t> lengths);

// Exact discrete water-filling: sum(b_i) == total_budget, min(2, n_i) <= b_i <= n_i.
std::vector<std::size_t> discrete_water_filling(std::span<const double> ell,
                                                std::span<const std::size_t> lengths,
                                                std::size_t total_budget,
                                                std::span<const float> log_anchors);

// Stage 2 MLP: predicts per-trajectory (w_geo, w_wa, w_rng) submodular weights.
std::vector<SubmodularWeights> predict_submodular_weights(std::span<const StudentModel> fold_models,
                                                          std::span<const std::uint32_t> fold_ids,
                                                          const TrajectoryFeatureTable &feats,
                                                          std::span<const std::size_t> lengths,
                                                          std::span<const std::size_t> budgets,
                                                          int rb_passes = 2);

} // namespace student

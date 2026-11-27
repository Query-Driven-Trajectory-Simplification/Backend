#include "backend/engine/student/student_model.hpp"

#include "backend/engine/core/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace student {

namespace {

constexpr std::uint32_t kModelMagic = 0x53544431U; // "STD1"
constexpr std::uint32_t kModelVersion = 1U;

template <typename T> T read_pod(std::istream &in) {
    T value{};
    if (!in.read(reinterpret_cast<char *>(&value), sizeof(T)))
        throw std::runtime_error("unexpected EOF reading student model");
    return value;
}

std::vector<float> read_floats(std::istream &in, std::size_t count) {
    std::vector<float> values(count);
    if (count > 0 && !in.read(reinterpret_cast<char *>(values.data()),
                              static_cast<std::streamsize>(count * sizeof(float))))
        throw std::runtime_error("unexpected EOF reading student model weights");
    return values;
}

std::vector<MlpLayer> read_layers(std::istream &in) {
    const auto num_layers = read_pod<std::uint32_t>(in);
    std::vector<MlpLayer> layers(num_layers);
    for (auto &layer : layers) {
        layer.in_dim = read_pod<std::uint32_t>(in);
        layer.out_dim = read_pod<std::uint32_t>(in);
        layer.has_layernorm = read_pod<std::uint32_t>(in) != 0;
        layer.weight = read_floats(in, static_cast<std::size_t>(layer.out_dim) * layer.in_dim);
        layer.bias = read_floats(in, layer.out_dim);
        if (layer.has_layernorm) {
            layer.ln_gamma = read_floats(in, layer.out_dim);
            layer.ln_beta = read_floats(in, layer.out_dim);
        }
    }
    return layers;
}

void run_mlp(std::span<const MlpLayer> layers, std::span<const float> input,
             std::vector<float> &activations, std::vector<float> &next_activations) {
    activations.assign(input.begin(), input.end());
    for (const auto &layer : layers) {
        next_activations.resize(layer.out_dim);
        for (std::uint32_t out_idx = 0; out_idx < layer.out_dim; ++out_idx) {
            const float *row_weights =
                layer.weight.data() + static_cast<std::size_t>(out_idx) * layer.in_dim;
            float sum = layer.bias[out_idx];
            for (std::uint32_t in_idx = 0; in_idx < layer.in_dim; ++in_idx)
                sum += row_weights[in_idx] * activations[in_idx];
            next_activations[out_idx] = sum;
        }
        if (layer.has_layernorm) {
            double mean = 0.0;
            for (float val : next_activations)
                mean += val;
            mean /= static_cast<double>(layer.out_dim);
            double variance = 0.0;
            for (float val : next_activations) {
                const double diff = val - mean;
                variance += diff * diff;
            }
            variance /= static_cast<double>(layer.out_dim);
            const double inv_std = 1.0 / std::sqrt(variance + 1e-5);
            for (std::uint32_t out_idx = 0; out_idx < layer.out_dim; ++out_idx) {
                const double normed = (next_activations[out_idx] - mean) * inv_std *
                                          layer.ln_gamma[out_idx] +
                                      layer.ln_beta[out_idx];
                next_activations[out_idx] =
                    static_cast<float>(normed / (1.0 + std::exp(-normed))); // SiLU
            }
        }
        activations.swap(next_activations);
    }
}

std::vector<double> anchor_spacings(std::span<const float> log_anchors) {
    std::vector<double> anchor_steps(log_anchors.size() - 1);
    for (std::size_t idx = 0; idx + 1 < log_anchors.size(); ++idx)
        anchor_steps[idx] =
            static_cast<double>(log_anchors[idx + 1]) - static_cast<double>(log_anchors[idx]);
    return anchor_steps;
}

// Continuous budget b_i(eta) obtained by inverting piecewise-linear log-marginal curve at level eta.
double continuous_budget_for_curve(const double *curve, std::size_t num_anchors,
                                   std::size_t traj_len, double eta,
                                   std::span<const float> log_anchors,
                                   std::span<const double> anchor_steps) {
    double log_budget = static_cast<double>(log_anchors[0]);
    for (std::size_t idx = 0; idx + 1 < num_anchors; ++idx) {
        const double frac =
            std::clamp((curve[idx] - eta) / (curve[idx] - curve[idx + 1]), 0.0, 1.0);
        log_budget += frac * anchor_steps[idx];
    }
    const double last_slope =
        (curve[num_anchors - 2] - curve[num_anchors - 1]) / anchor_steps.back();
    const double tail = std::clamp((curve[num_anchors - 1] - eta) / (last_slope + 1e-3), 0.0, 4.0);
    return std::clamp(std::exp(log_budget + tail),
                      static_cast<double>(std::min<std::size_t>(2, traj_len)),
                      static_cast<double>(traj_len));
}

// Binary search for the unique dual price eta* where sum_{i in indices} b_i(eta*) == target_budget.
double bisect_water_level(std::span<const double> marginal_curves,
                          std::span<const std::size_t> lengths,
                          std::span<const std::size_t> indices, double target_budget,
                          std::span<const float> log_anchors,
                          std::span<const double> anchor_steps) {
    const std::size_t num_anchors = log_anchors.size();
    double low_eta = std::numeric_limits<double>::infinity();
    double high_eta = -std::numeric_limits<double>::infinity();
    for (std::size_t traj_idx : indices) {
        for (std::size_t anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx) {
            const double val = marginal_curves[traj_idx * num_anchors + anchor_idx];
            if (val < low_eta)
                low_eta = val;
            if (val > high_eta)
                high_eta = val;
        }
    }
    low_eta -= 15.0;
    high_eta += 2.0;
    for (int iter = 0; iter < 44; ++iter) {
        const double mid_eta = 0.5 * (low_eta + high_eta);
        double budget_sum = 0.0;
        for (std::size_t traj_idx : indices) {
            budget_sum += continuous_budget_for_curve(
                marginal_curves.data() + traj_idx * num_anchors, num_anchors, lengths[traj_idx],
                mid_eta, log_anchors, anchor_steps);
        }
        if (budget_sum > target_budget)
            low_eta = mid_eta;
        else
            high_eta = mid_eta;
    }
    return 0.5 * (low_eta + high_eta);
}

} // namespace

StudentModel load_model(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open student model: " + path.string());
    if (read_pod<std::uint32_t>(in) != kModelMagic || read_pod<std::uint32_t>(in) != kModelVersion)
        throw std::runtime_error("invalid student model header: " + path.string());

    StudentModel model;
    model.s1_in_dim = read_pod<std::uint32_t>(in);
    model.num_anchors = read_pod<std::uint32_t>(in);
    model.s1_mean = read_floats(in, model.s1_in_dim);
    model.s1_std = read_floats(in, model.s1_in_dim);
    model.log_anchors = read_floats(in, model.num_anchors);
    model.s1_layers = read_layers(in);

    model.s2_in_dim = read_pod<std::uint32_t>(in);
    model.num_presets = read_pod<std::uint32_t>(in);
    model.s2_mean = read_floats(in, model.s2_in_dim);
    model.s2_std = read_floats(in, model.s2_in_dim);
    model.log_simplex = read_floats(in, static_cast<std::size_t>(model.num_presets) * 3);
    model.s2_layers = read_layers(in);
    return model;
}

std::vector<double> predict_marginal_curves(std::span<const StudentModel> fold_models,
                                            std::span<const std::uint32_t> fold_ids,
                                            const TrajectoryFeatureTable &feats,
                                            std::span<const std::size_t> lengths) {
    const std::size_t num_trajs = feats.num_trajs;
    const std::size_t num_features = fold_models[0].s1_in_dim;
    const std::size_t num_anchors = fold_models[0].num_anchors;
    const bool is_multifold = fold_models.size() > 1;
    if (feats.feature_names.size() != num_features || (is_multifold && fold_ids.size() != num_trajs))
        throw std::invalid_argument("dimension mismatch in predict_marginal_curves");

    std::vector<double> marginal_curves(num_trajs * num_anchors, 0.0);
    core::parallel_for(num_trajs, [&](std::size_t traj_idx) {
        const auto &model = fold_models[is_multifold ? fold_ids[traj_idx] : 0];
        std::vector<float> input(num_features), activations, scratch;
        const float *feature_row = feats.values.data() + traj_idx * num_features;
        for (std::size_t feat_idx = 0; feat_idx < num_features; ++feat_idx)
            input[feat_idx] = (feature_row[feat_idx] - model.s1_mean[feat_idx]) / model.s1_std[feat_idx];
        run_mlp(model.s1_layers, input, activations, scratch);

        double current_val = static_cast<double>(activations[0]);
        double *curve_out = marginal_curves.data() + traj_idx * num_anchors;
        curve_out[0] = current_val;
        for (std::size_t anchor_idx = 1; anchor_idx < num_anchors; ++anchor_idx) {
            const double raw_step = static_cast<double>(activations[anchor_idx]);
            current_val -=
                ((raw_step > 20.0 ? raw_step : std::log1p(std::exp(raw_step))) + 1e-2);
            curve_out[anchor_idx] = current_val;
        }
    });

    // Align fold dual prices at 1% reference share before merging into one global water-fill
    if (is_multifold) {
        const auto anchor_steps = anchor_spacings(fold_models[0].log_anchors);
        for (std::size_t fold = 0; fold < fold_models.size(); ++fold) {
            std::vector<std::size_t> fold_trajs;
            std::size_t total_points = 0, total_endpoints = 0;
            for (std::size_t traj_idx = 0; traj_idx < num_trajs; ++traj_idx) {
                if (fold_ids[traj_idx] == fold) {
                    fold_trajs.push_back(traj_idx);
                    total_points += lengths[traj_idx];
                    total_endpoints += std::min<std::size_t>(2, lengths[traj_idx]);
                }
            }
            const double ref_budget =
                static_cast<double>(total_endpoints) +
                0.01 * static_cast<double>(total_points - total_endpoints);
            const double eta_ref = bisect_water_level(marginal_curves, lengths, fold_trajs,
                                                      ref_budget, fold_models[0].log_anchors,
                                                      anchor_steps);
            for (std::size_t traj_idx : fold_trajs)
                for (std::size_t anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx)
                    marginal_curves[traj_idx * num_anchors + anchor_idx] -= eta_ref;
        }
    }
    return marginal_curves;
}

std::vector<std::size_t> discrete_water_filling(std::span<const double> marginal_curves,
                                                std::span<const std::size_t> lengths,
                                                std::size_t total_budget,
                                                std::span<const float> log_anchors) {
    const std::size_t num_trajs = lengths.size();
    const std::size_t num_anchors = log_anchors.size();
    const auto anchor_steps = anchor_spacings(log_anchors);

    std::vector<std::size_t> all_indices(num_trajs);
    for (std::size_t traj_idx = 0; traj_idx < num_trajs; ++traj_idx)
        all_indices[traj_idx] = traj_idx;
    const double eta_star = bisect_water_level(marginal_curves, lengths, all_indices,
                                               static_cast<double>(total_budget), log_anchors,
                                               anchor_steps);

    std::vector<std::size_t> budgets(num_trajs);
    std::size_t current_sum = 0;
    for (std::size_t traj_idx = 0; traj_idx < num_trajs; ++traj_idx) {
        const std::size_t min_budget = std::min<std::size_t>(2, lengths[traj_idx]);
        const double continuous_budget = continuous_budget_for_curve(
            marginal_curves.data() + traj_idx * num_anchors, num_anchors, lengths[traj_idx],
            eta_star, log_anchors, anchor_steps);
        budgets[traj_idx] = std::clamp(static_cast<std::size_t>(std::floor(continuous_budget)),
                                       min_budget, lengths[traj_idx]);
        current_sum += budgets[traj_idx];
    }

    auto eval_marginal_at = [&](std::size_t traj_idx, std::size_t budget_val) {
        const double log_budget = std::log(std::max(2.0, static_cast<double>(budget_val)));
        auto it =
            std::upper_bound(log_anchors.begin(), log_anchors.end(), static_cast<float>(log_budget));
        std::size_t anchor_idx = (it == log_anchors.begin())
                                     ? 0
                                     : static_cast<std::size_t>(it - log_anchors.begin() - 1);
        if (anchor_idx >= num_anchors - 1)
            anchor_idx = num_anchors - 2;
        const double frac =
            (log_budget - static_cast<double>(log_anchors[anchor_idx])) / anchor_steps[anchor_idx];
        const std::size_t base = traj_idx * num_anchors + anchor_idx;
        return marginal_curves[base] + frac * (marginal_curves[base + 1] - marginal_curves[base]);
    };

    std::vector<std::pair<double, std::size_t>> candidates;
    candidates.reserve(num_trajs);
    while (current_sum < total_budget) {
        candidates.clear();
        for (std::size_t traj_idx = 0; traj_idx < num_trajs; ++traj_idx)
            if (budgets[traj_idx] < lengths[traj_idx])
                candidates.emplace_back(eval_marginal_at(traj_idx, budgets[traj_idx] + 1), traj_idx);
        const std::size_t num_add = std::min(total_budget - current_sum, candidates.size());
        if (num_add == 0)
            break;
        std::nth_element(candidates.begin(),
                         candidates.begin() + static_cast<std::ptrdiff_t>(num_add), candidates.end(),
                         [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
        for (std::size_t k = 0; k < num_add; ++k)
            budgets[candidates[k].second] += 1;
        current_sum += num_add;
    }
    while (current_sum > total_budget) {
        candidates.clear();
        for (std::size_t traj_idx = 0; traj_idx < num_trajs; ++traj_idx)
            if (budgets[traj_idx] > std::min<std::size_t>(2, lengths[traj_idx]))
                candidates.emplace_back(eval_marginal_at(traj_idx, budgets[traj_idx]), traj_idx);
        const std::size_t num_remove = std::min(current_sum - total_budget, candidates.size());
        if (num_remove == 0)
            break;
        std::nth_element(candidates.begin(),
                         candidates.begin() + static_cast<std::ptrdiff_t>(num_remove),
                         candidates.end(),
                         [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
        for (std::size_t k = 0; k < num_remove; ++k)
            budgets[candidates[k].second] -= 1;
        current_sum -= num_remove;
    }
    if (current_sum != total_budget)
        throw std::runtime_error("discrete_water_filling could not satisfy total_budget");
    return budgets;
}

std::vector<SubmodularWeights> predict_submodular_weights(std::span<const StudentModel> fold_models,
                                                          std::span<const std::uint32_t> fold_ids,
                                                          const TrajectoryFeatureTable &feats,
                                                          std::span<const std::size_t> lengths,
                                                          std::span<const std::size_t> budgets,
                                                          int rb_passes) {
    const std::size_t num_trajs = feats.num_trajs;
    const std::size_t num_features = fold_models[0].s1_in_dim;
    const std::size_t num_presets = fold_models[0].num_presets;
    const bool is_multifold = fold_models.size() > 1;
    if (feats.feature_names.size() + 3 != fold_models[0].s2_in_dim ||
        (is_multifold && fold_ids.size() != num_trajs))
        throw std::invalid_argument("dimension mismatch in predict_submodular_weights");

    std::vector<SubmodularWeights> weights_out(num_trajs);
    core::parallel_for(num_trajs, [&](std::size_t traj_idx) {
        const auto &model = fold_models[is_multifold ? fold_ids[traj_idx] : 0];
        std::vector<float> input(num_features + 3), activations, scratch;
        const float *feature_row = feats.values.data() + traj_idx * num_features;
        for (std::size_t feat_idx = 0; feat_idx < num_features; ++feat_idx)
            input[feat_idx] = feature_row[feat_idx];
        const float budget_f = std::max(2.0f, static_cast<float>(budgets[traj_idx]));
        const float length_f = std::max(2.0f, static_cast<float>(lengths[traj_idx]));
        input[num_features + 0] = std::log(budget_f);
        input[num_features + 1] = budget_f / length_f;
        input[num_features + 2] = std::log(length_f / budget_f);
        for (std::size_t feat_idx = 0; feat_idx < num_features + 3; ++feat_idx)
            input[feat_idx] = (input[feat_idx] - model.s2_mean[feat_idx]) / model.s2_std[feat_idx];
        run_mlp(model.s2_layers, input, activations, scratch);

        double max_logit = -std::numeric_limits<double>::infinity();
        for (std::size_t preset_idx = 0; preset_idx < num_presets; ++preset_idx)
            max_logit = std::max(max_logit, static_cast<double>(activations[preset_idx]));
        double exp_sum = 0.0;
        double log_weights[3] = {0.0, 0.0, 0.0};
        for (std::size_t preset_idx = 0; preset_idx < num_presets; ++preset_idx) {
            const double weight_exp =
                std::exp(static_cast<double>(activations[preset_idx]) - max_logit);
            exp_sum += weight_exp;
            log_weights[0] += weight_exp * model.log_simplex[preset_idx * 3 + 0];
            log_weights[1] += weight_exp * model.log_simplex[preset_idx * 3 + 1];
            log_weights[2] += weight_exp * model.log_simplex[preset_idx * 3 + 2];
        }
        weights_out[traj_idx] = {std::exp(log_weights[0] / exp_sum),
                                 std::exp(log_weights[1] / exp_sum),
                                 std::exp(log_weights[2] / exp_sum), rb_passes};
    });
    return weights_out;
}

} // namespace student

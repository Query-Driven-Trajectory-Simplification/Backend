#include "backend/engine/student/student_model.hpp"

#include "backend/engine/core/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace student {

namespace {

constexpr std::uint32_t kModelMagic = 0x53544431U; // "STD1"
constexpr std::uint32_t kModelVersion = 1U;

template <typename T> T read_pod(std::istream &in) {
    T v{};
    if (!in.read(reinterpret_cast<char *>(&v), sizeof(T)))
        throw std::runtime_error("unexpected EOF reading student model");
    return v;
}

std::vector<float> read_floats(std::istream &in, std::size_t count) {
    std::vector<float> out(count);
    if (count > 0 && !in.read(reinterpret_cast<char *>(out.data()),
                              static_cast<std::streamsize>(count * sizeof(float))))
        throw std::runtime_error("unexpected EOF reading student model weights");
    return out;
}

std::vector<MlpLayer> read_layers(std::istream &in) {
    const auto num_layers = read_pod<std::uint32_t>(in);
    std::vector<MlpLayer> layers(num_layers);
    for (auto &L : layers) {
        L.in_dim = read_pod<std::uint32_t>(in);
        L.out_dim = read_pod<std::uint32_t>(in);
        L.has_layernorm = read_pod<std::uint32_t>(in) != 0;
        L.weight = read_floats(in, static_cast<std::size_t>(L.out_dim) * L.in_dim);
        L.bias = read_floats(in, L.out_dim);
        if (L.has_layernorm) {
            L.ln_gamma = read_floats(in, L.out_dim);
            L.ln_beta = read_floats(in, L.out_dim);
        }
    }
    return layers;
}

void run_mlp(std::span<const MlpLayer> layers, std::span<const float> x,
             std::vector<float> &buf_a, std::vector<float> &buf_b) {
    buf_a.assign(x.begin(), x.end());
    for (const auto &L : layers) {
        buf_b.resize(L.out_dim);
        for (std::uint32_t o = 0; o < L.out_dim; ++o) {
            const float *w = L.weight.data() + static_cast<std::size_t>(o) * L.in_dim;
            float acc = L.bias[o];
            for (std::uint32_t j = 0; j < L.in_dim; ++j)
                acc += w[j] * buf_a[j];
            buf_b[o] = acc;
        }
        if (L.has_layernorm) {
            double mean = 0.0;
            for (float v : buf_b)
                mean += v;
            mean /= static_cast<double>(L.out_dim);
            double var = 0.0;
            for (float v : buf_b) {
                const double d = v - mean;
                var += d * d;
            }
            var /= static_cast<double>(L.out_dim);
            const double inv_std = 1.0 / std::sqrt(var + 1e-5);
            for (std::uint32_t o = 0; o < L.out_dim; ++o) {
                const double z = (buf_b[o] - mean) * inv_std * L.ln_gamma[o] + L.ln_beta[o];
                buf_b[o] = static_cast<float>(z / (1.0 + std::exp(-z))); // SiLU
            }
        }
        buf_a.swap(buf_b);
    }
}

std::vector<double> anchor_spacings(std::span<const float> log_anchors) {
    std::vector<double> du(log_anchors.size() - 1);
    for (std::size_t m = 0; m + 1 < log_anchors.size(); ++m)
        du[m] = static_cast<double>(log_anchors[m + 1]) - static_cast<double>(log_anchors[m]);
    return du;
}

// Continuous budget b_i(eta) obtained by inverting piecewise-linear log-marginal curve row at level eta.
double cont_budget_row(const double *row, std::size_t M, std::size_t len, double eta,
                       std::span<const float> log_anchors, std::span<const double> du) {
    double log_b = static_cast<double>(log_anchors[0]);
    for (std::size_t m = 0; m + 1 < M; ++m)
        log_b += std::clamp((row[m] - eta) / (row[m] - row[m + 1]), 0.0, 1.0) * du[m];
    const double last_slope = (row[M - 2] - row[M - 1]) / du.back();
    const double tail = std::clamp((row[M - 1] - eta) / (last_slope + 1e-3), 0.0, 4.0);
    return std::clamp(std::exp(log_b + tail),
                      static_cast<double>(std::min<std::size_t>(2, len)),
                      static_cast<double>(len));
}

// Binary search for the unique dual price eta* where sum_{i in indices} b_i(eta*) == target_budget.
double bisect_water_level(std::span<const double> ell, std::span<const std::size_t> lengths,
                          std::span<const std::size_t> indices, double target_budget,
                          std::span<const float> log_anchors, std::span<const double> du) {
    const std::size_t M = log_anchors.size();
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (std::size_t i : indices) {
        for (std::size_t m = 0; m < M; ++m) {
            const double v = ell[i * M + m];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
    }
    lo -= 15.0;
    hi += 2.0;
    for (int it = 0; it < 44; ++it) {
        const double mid = 0.5 * (lo + hi);
        double sum = 0.0;
        for (std::size_t i : indices)
            sum += cont_budget_row(ell.data() + i * M, M, lengths[i], mid, log_anchors, du);
        if (sum > target_budget) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

} // namespace

StudentModel load_model(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open student model: " + path.string());
    if (read_pod<std::uint32_t>(in) != kModelMagic || read_pod<std::uint32_t>(in) != kModelVersion)
        throw std::runtime_error("invalid student model header: " + path.string());

    StudentModel m;
    m.s1_in_dim = read_pod<std::uint32_t>(in);
    m.num_anchors = read_pod<std::uint32_t>(in);
    m.s1_mean = read_floats(in, m.s1_in_dim);
    m.s1_std = read_floats(in, m.s1_in_dim);
    m.log_anchors = read_floats(in, m.num_anchors);
    m.s1_layers = read_layers(in);

    m.s2_in_dim = read_pod<std::uint32_t>(in);
    m.num_presets = read_pod<std::uint32_t>(in);
    m.s2_mean = read_floats(in, m.s2_in_dim);
    m.s2_std = read_floats(in, m.s2_in_dim);
    m.log_simplex = read_floats(in, static_cast<std::size_t>(m.num_presets) * 3);
    m.s2_layers = read_layers(in);
    return m;
}

std::vector<double> predict_marginal_curves(std::span<const StudentModel> fold_models,
                                            std::span<const std::uint32_t> fold_ids,
                                            const TrajectoryFeatureTable &feats,
                                            std::span<const std::size_t> lengths) {
    const std::size_t N = feats.num_trajs;
    const std::size_t F = fold_models[0].s1_in_dim;
    const std::size_t M = fold_models[0].num_anchors;
    const bool multi = fold_models.size() > 1;
    if (feats.feature_names.size() != F || (multi && fold_ids.size() != N))
        throw std::invalid_argument("dimension mismatch in predict_marginal_curves");

    std::vector<double> ell(N * M, 0.0);
    core::parallel_for(N, [&](std::size_t i) {
        const auto &m = fold_models[multi ? fold_ids[i] : 0];
        std::vector<float> x(F), buf_a, buf_b;
        const float *row = feats.values.data() + i * F;
        for (std::size_t j = 0; j < F; ++j)
            x[j] = (row[j] - m.s1_mean[j]) / m.s1_std[j];
        run_mlp(m.s1_layers, x, buf_a, buf_b);

        double cur = static_cast<double>(buf_a[0]);
        double *out = ell.data() + i * M;
        out[0] = cur;
        for (std::size_t k = 1; k < M; ++k) {
            const double z = static_cast<double>(buf_a[k]);
            cur -= ((z > 20.0 ? z : std::log1p(std::exp(z))) + 1e-2);
            out[k] = cur;
        }
    });

    // Align fold dual prices at 1% reference share before merging into one global water-fill
    if (multi) {
        const auto du = anchor_spacings(fold_models[0].log_anchors);
        for (std::size_t k = 0; k < fold_models.size(); ++k) {
            std::vector<std::size_t> idx;
            std::size_t pts = 0, eps = 0;
            for (std::size_t i = 0; i < N; ++i) {
                if (fold_ids[i] == k) {
                    idx.push_back(i);
                    pts += lengths[i];
                    eps += std::min<std::size_t>(2, lengths[i]);
                }
            }
            const double b_ref = static_cast<double>(eps) + 0.01 * static_cast<double>(pts - eps);
            const double eta_ref =
                bisect_water_level(ell, lengths, idx, b_ref, fold_models[0].log_anchors, du);
            for (std::size_t i : idx)
                for (std::size_t m = 0; m < M; ++m)
                    ell[i * M + m] -= eta_ref;
        }
    }
    return ell;
}

std::vector<std::size_t> discrete_water_filling(std::span<const double> ell,
                                                std::span<const std::size_t> lengths,
                                                std::size_t total_budget,
                                                std::span<const float> log_anchors) {
    const std::size_t N = lengths.size(), M = log_anchors.size();
    const auto du = anchor_spacings(log_anchors);

    std::vector<std::size_t> all_idx(N);
    for (std::size_t i = 0; i < N; ++i) all_idx[i] = i;
    const double eta_star = bisect_water_level(
        ell, lengths, all_idx, static_cast<double>(total_budget), log_anchors, du);

    std::vector<std::size_t> b(N);
    std::size_t cur_sum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t min_b = std::min<std::size_t>(2, lengths[i]);
        const double cb = cont_budget_row(ell.data() + i * M, M, lengths[i], eta_star, log_anchors, du);
        b[i] = std::clamp(static_cast<std::size_t>(std::floor(cb)), min_b, lengths[i]);
        cur_sum += b[i];
    }

    auto eval_ell_at = [&](std::size_t i, std::size_t bv) {
        const double lb = std::log(std::max(2.0, static_cast<double>(bv)));
        auto it = std::upper_bound(log_anchors.begin(), log_anchors.end(), static_cast<float>(lb));
        std::size_t idx = (it == log_anchors.begin()) ? 0 : static_cast<std::size_t>(it - log_anchors.begin() - 1);
        if (idx >= M - 1) idx = M - 2;
        const double t = (lb - static_cast<double>(log_anchors[idx])) / du[idx];
        return ell[i * M + idx] + t * (ell[i * M + idx + 1] - ell[i * M + idx]);
    };

    std::vector<std::pair<double, std::size_t>> cand;
    cand.reserve(N);
    while (cur_sum < total_budget) {
        cand.clear();
        for (std::size_t i = 0; i < N; ++i)
            if (b[i] < lengths[i]) cand.emplace_back(eval_ell_at(i, b[i] + 1), i);
        const std::size_t take = std::min(total_budget - cur_sum, cand.size());
        if (take == 0) break;
        std::nth_element(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(take), cand.end(),
                         [](const auto &a, const auto &c) { return a.first > c.first; });
        for (std::size_t k = 0; k < take; ++k) b[cand[k].second] += 1;
        cur_sum += take;
    }
    while (cur_sum > total_budget) {
        cand.clear();
        for (std::size_t i = 0; i < N; ++i)
            if (b[i] > std::min<std::size_t>(2, lengths[i])) cand.emplace_back(eval_ell_at(i, b[i]), i);
        const std::size_t drop = std::min(cur_sum - total_budget, cand.size());
        if (drop == 0) break;
        std::nth_element(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(drop), cand.end(),
                         [](const auto &a, const auto &c) { return a.first < c.first; });
        for (std::size_t k = 0; k < drop; ++k) b[cand[k].second] -= 1;
        cur_sum -= drop;
    }
    if (cur_sum != total_budget)
        throw std::runtime_error("discrete_water_filling could not satisfy total_budget");
    return b;
}

std::vector<SubmodularWeights> predict_submodular_weights(std::span<const StudentModel> fold_models,
                                                          std::span<const std::uint32_t> fold_ids,
                                                          const TrajectoryFeatureTable &feats,
                                                          std::span<const std::size_t> lengths,
                                                          std::span<const std::size_t> budgets,
                                                          int rb_passes) {
    const std::size_t N = feats.num_trajs;
    const std::size_t F = fold_models[0].s1_in_dim;
    const std::size_t K = fold_models[0].num_presets;
    const bool multi = fold_models.size() > 1;
    if (feats.feature_names.size() + 3 != fold_models[0].s2_in_dim || (multi && fold_ids.size() != N))
        throw std::invalid_argument("dimension mismatch in predict_submodular_weights");

    std::vector<SubmodularWeights> out(N);
    core::parallel_for(N, [&](std::size_t i) {
        const auto &m = fold_models[multi ? fold_ids[i] : 0];
        std::vector<float> x(F + 3), buf_a, buf_b;
        const float *row = feats.values.data() + i * F;
        for (std::size_t j = 0; j < F; ++j) x[j] = row[j];
        const float bf = std::max(2.0f, static_cast<float>(budgets[i]));
        const float nf = std::max(2.0f, static_cast<float>(lengths[i]));
        x[F + 0] = std::log(bf);
        x[F + 1] = bf / nf;
        x[F + 2] = std::log(nf / bf);
        for (std::size_t j = 0; j < F + 3; ++j)
            x[j] = (x[j] - m.s2_mean[j]) / m.s2_std[j];
        run_mlp(m.s2_layers, x, buf_a, buf_b);

        double max_l = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < K; ++k)
            max_l = std::max(max_l, static_cast<double>(buf_a[k]));
        double sum_e = 0.0, lw[3] = {0.0, 0.0, 0.0};
        for (std::size_t k = 0; k < K; ++k) {
            const double e = std::exp(static_cast<double>(buf_a[k]) - max_l);
            sum_e += e;
            lw[0] += e * m.log_simplex[k * 3 + 0];
            lw[1] += e * m.log_simplex[k * 3 + 1];
            lw[2] += e * m.log_simplex[k * 3 + 2];
        }
        out[i] = {std::exp(lw[0] / sum_e), std::exp(lw[1] / sum_e), std::exp(lw[2] / sum_e),
                  rb_passes};
    });
    return out;
}

} // namespace student

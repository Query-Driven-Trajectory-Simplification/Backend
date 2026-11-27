#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/timer.hpp"
#include "backend/engine/evaluator/combined.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/join.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/pipeline/test_workloads.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/selection/distance_functions.hpp"
#include "backend/engine/selection/rdp.hpp"
#include "backend/engine/selection/sed.hpp"
#include "backend/engine/split/split.hpp"
#include "backend/engine/student/candidate_pool.hpp"
#include "backend/engine/student/density_index.hpp"
#include "backend/engine/student/features.hpp"
#include "backend/engine/student/range_sampler.hpp"
#include "backend/engine/student/student.hpp"
#include "backend/engine/student/submodular_solver.hpp"
#include "backend/engine/teacher/teacher.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
void write_raw(const std::filesystem::path &path, const std::vector<T> &values) {
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot open for write: " + path.string());
    if (!values.empty())
        out.write(reinterpret_cast<const char *>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
}

template <typename T> std::vector<T> read_raw(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("cannot open for read: " + path.string());
    const auto byte_size = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    std::vector<T> values(byte_size / sizeof(T));
    if (!values.empty())
        in.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(byte_size));
    return values;
}

void save_kept_binary(const std::filesystem::path &path, const core::KeptSets &kept) {
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t num_trajs = kept.size();
    out.write(reinterpret_cast<const char *>(&num_trajs), sizeof(num_trajs));
    for (const auto &traj_kept : kept) {
        const auto count = static_cast<std::uint32_t>(traj_kept.size());
        out.write(reinterpret_cast<const char *>(&count), sizeof(count));
        for (std::size_t idx : traj_kept) {
            const auto pt_idx = static_cast<std::uint32_t>(idx);
            out.write(reinterpret_cast<const char *>(&pt_idx), sizeof(pt_idx));
        }
    }
}

core::KeptSets load_kept_binary(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open kept binary: " + path.string());
    std::uint64_t num_trajs = 0;
    in.read(reinterpret_cast<char *>(&num_trajs), sizeof(num_trajs));
    core::KeptSets kept(num_trajs);
    for (std::uint64_t i = 0; i < num_trajs; ++i) {
        std::uint32_t count = 0;
        in.read(reinterpret_cast<char *>(&count), sizeof(count));
        kept[i].resize(count);
        for (std::uint32_t j = 0; j < count; ++j) {
            std::uint32_t pt_idx = 0;
            in.read(reinterpret_cast<char *>(&pt_idx), sizeof(pt_idx));
            kept[i][j] = pt_idx;
        }
    }
    return kept;
}

struct CliArgs {
    std::string cmd;
    std::filesystem::path config_path = "engine/configs/geolife.json";
    std::string split = "full"; // full | half0 | half1
    std::optional<std::uint64_t> seed_override;
    std::optional<std::uint64_t> pseudo_split;
    std::filesystem::path out_dir = "results/student_export";
    bool baselines = false;
    std::vector<std::filesystem::path> model_paths; // 1 path = single model; 2 = 2-fold OOF
    std::optional<std::filesystem::path> teacher_dir;
    std::optional<std::filesystem::path> csv_path;
};

CliArgs parse_args(int argc, char **argv) {
    CliArgs args;
    args.cmd = argv[1];
    for (int idx = 2; idx < argc; ++idx) {
        const std::string flag = argv[idx];
        auto next_value = [&] {
            if (idx + 1 >= argc)
                throw std::invalid_argument("missing value for " + flag);
            return std::string(argv[++idx]);
        };
        if (flag == "--config")
            args.config_path = next_value();
        else if (flag == "--split")
            args.split = next_value();
        else if (flag == "--seed")
            args.seed_override = std::stoull(next_value());
        else if (flag == "--pseudo-split")
            args.pseudo_split = std::stoull(next_value());
        else if (flag == "--out-dir")
            args.out_dir = next_value();
        else if (flag == "--baselines")
            args.baselines = true;
        else if (flag == "--model") {
            std::stringstream ss(next_value());
            for (std::string item; std::getline(ss, item, ',');)
                args.model_paths.emplace_back(item);
        } else if (flag == "--teacher-dir")
            args.teacher_dir = next_value();
        else if (flag == "--csv")
            args.csv_path = next_value();
        else
            throw std::invalid_argument("unknown flag: " + flag);
    }
    return args;
}

struct SplitData {
    core::TrajectoryDb db;
    core::TrajectoryDb pool;
    std::filesystem::path cache_dir;
    std::vector<std::uint32_t> fold_assignment;
};

SplitData load_split(const pipeline::Config &config, const CliArgs &args, std::uint64_t seed) {
    const auto all_trajs = io::load_trajectories(config.data_path);
    const auto outer = split::split(all_trajs, config.split, seed);

    core::TrajectoryDb base_db = core::subset(all_trajs, outer.database);
    core::TrajectoryDb base_pool = core::subset(all_trajs, outer.queries);
    std::filesystem::path cache_dir = config.teacher.cache_dir;

    if (args.pseudo_split) {
        const auto inner = split::split(base_db, config.split, *args.pseudo_split);
        auto inner_db = core::subset(base_db, inner.database);
        auto inner_pool = core::subset(base_db, inner.queries);
        base_db = std::move(inner_db);
        base_pool = std::move(inner_pool);
        cache_dir = config.teacher.cache_dir / std::format("pseudo_{}", *args.pseudo_split);
    }

    std::vector<std::size_t> perm(base_db.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937_64 rng(seed * 1315423911ULL + (args.pseudo_split.value_or(0) * 2654435761ULL) +
                        17);
    // Portable Fisher-Yates shuffle (identical across libstdc++, libc++, and MSVC)
    for (std::size_t i = perm.size(); i > 1; --i)
        std::swap(perm[i - 1], perm[static_cast<std::size_t>(rng() % i)]);

    const std::size_t midpoint = perm.size() / 2;
    std::vector<std::uint32_t> fold_of(base_db.size(), 0);
    for (std::size_t pos = 0; pos < perm.size(); ++pos)
        fold_of[perm[pos]] = (pos < midpoint) ? 0U : 1U;

    if (args.split == "half0" || args.split == "half1") {
        const int half = (args.split == "half1") ? 1 : 0;
        std::vector<std::size_t> indices =
            (half == 0)
                ? std::vector<std::size_t>(perm.begin(),
                                           perm.begin() + static_cast<std::ptrdiff_t>(midpoint))
                : std::vector<std::size_t>(perm.begin() + static_cast<std::ptrdiff_t>(midpoint),
                                           perm.end());
        std::ranges::sort(indices);
        return {core::subset(base_db, indices), std::move(base_pool),
                cache_dir / std::format("half_{}", half), {}};
    }

    if (args.split != "full")
        throw std::invalid_argument("unknown --split (expected full, half0, half1): " + args.split);
    return {std::move(base_db), std::move(base_pool), std::move(cache_dir), std::move(fold_of)};
}

void validate_kept(const core::KeptSets &kept, const core::TrajectoryDb &db,
                   std::size_t expected_budget) {
    if (kept.size() != db.size())
        throw std::runtime_error("kept size mismatch");
    std::size_t total = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto traj_len = db.trajectory(i).size();
        const auto &traj_kept = kept[i];
        if (traj_kept.size() < std::min<std::size_t>(2, traj_len) || traj_kept.size() > traj_len)
            throw std::runtime_error(
                std::format("traj {} invalid budget {} (n={})", i, traj_kept.size(), traj_len));
        if (traj_kept.front() != 0 || traj_kept.back() != traj_len - 1)
            throw std::runtime_error(std::format("traj {} missing endpoints", i));
        for (std::size_t j = 1; j < traj_kept.size(); ++j)
            if (traj_kept[j] <= traj_kept[j - 1])
                throw std::runtime_error(std::format("traj {} non-increasing indices", i));
        total += traj_kept.size();
    }
    if (total != expected_budget)
        throw std::runtime_error(
            std::format("budget mismatch: got {} expected {}", total, expected_budget));
}

// =============================================================================
// 1. EXPORT: 13 zero-Frechet features + cached frozen Teacher targets
// =============================================================================
void run_export(const core::TrajectoryDb &db, const pipeline::Config &config,
                const SplitData &split_data, const CliArgs &args, std::uint64_t seed,
                std::span<const double> shares) {
    std::filesystem::create_directories(args.out_dir);

    auto t_density = std::chrono::steady_clock::now();
    const student::DbDensityIndex density_idx(db);
    const auto range_idx = student::build_sampled_range_index(db, 128000);
    const double density_ms = core::seconds_since(t_density) * 1000.0;

    auto t_features = std::chrono::steady_clock::now();
    const auto features = student::extract_trajectory_features(db, density_idx, &range_idx);
    const double feature_ms = core::seconds_since(t_features) * 1000.0;

    std::vector<std::uint32_t> lengths(db.size());
    for (std::size_t i = 0; i < db.size(); ++i)
        lengths[i] = static_cast<std::uint32_t>(db.trajectory(i).size());

    write_raw(args.out_dir / "features.bin", features.values);
    write_raw(args.out_dir / "lengths.bin", lengths);

    nlohmann::json meta{
        {"num_trajs", features.num_trajs},
        {"num_points", db.num_points()},
        {"num_features", features.feature_names.size()},
        {"feature_names", features.feature_names},
        {"shares", std::vector<double>(shares.begin(), shares.end())},
        {"density_ms", density_ms},
        {"feature_ms", feature_ms},
    };
    std::ofstream(args.out_dir / "meta.json") << meta.dump(2) << '\n';
    std::cerr << std::format("Exported {} trajs x {} feats in {:.1f}ms dens + {:.1f}ms feat\n",
                             features.num_trajs, features.feature_names.size(), density_ms,
                             feature_ms);

    bool any_missing = false;
    for (double share : shares)
        if (!std::filesystem::exists(args.out_dir / std::format("teacher_kept_{:g}.bin", share)))
            any_missing = true;
    if (!any_missing)
        return;

    std::filesystem::create_directories(split_data.cache_dir);
    auto t_setup = std::chrono::steady_clock::now();
    auto workloads = teacher::load_or_build_workloads(
        db, pipeline::teacher_workload_config(config), seed, split_data.cache_dir);
    if (config.teacher.term_weights[teacher::Objective::Margin] > 0)
        teacher::add_ranked_neighbours(db, workloads, teacher::kMarginOutside, seed,
                                       split_data.cache_dir);
    const teacher::Teacher teacher_model(db, std::move(workloads),
                                         pipeline::teacher_objective_config(config), seed);
    std::cerr << std::format("Teacher setup in {:.2f}s\n", core::seconds_since(t_setup));

    const std::vector<std::string> arms{"BUDGET_RDP", "BUDGET_LOOP_REFINE"};
    for (double share : shares) {
        const auto kept_path = args.out_dir / std::format("teacher_kept_{:g}.bin", share);
        if (std::filesystem::exists(kept_path))
            continue;
        auto t_share = std::chrono::steady_clock::now();
        auto teacher_results = teacher_model.run(allocation::total_budget(db, share), arms);
        auto rdp_kept = core::to_kept(teacher_results[0].simplification);
        auto refine_kept = core::to_kept(teacher_results[1].simplification);

        std::vector<std::uint32_t> rdp_budgets(db.size()), refine_budgets(db.size());
        for (std::size_t i = 0; i < db.size(); ++i) {
            rdp_budgets[i] = static_cast<std::uint32_t>(rdp_kept[i].size());
            refine_budgets[i] = static_cast<std::uint32_t>(refine_kept[i].size());
        }
        write_raw(args.out_dir / std::format("teacher_rdp_budgets_{:g}.bin", share), rdp_budgets);
        write_raw(args.out_dir / std::format("teacher_budgets_{:g}.bin", share), refine_budgets);
        save_kept_binary(kept_path, refine_kept);
        std::cerr << std::format("  share {:g} Teacher saved in {:.2f}s\n", share,
                                 core::seconds_since(t_share));
    }
}

// =============================================================================
// 2. FIT-SIMPLEX: offline inverse-optimization targets for Stage 2 MLP
// =============================================================================
void run_fit_simplex(const core::TrajectoryDb &db, const CliArgs &args,
                     std::span<const double> shares) {
    const std::vector<std::array<double, 3>> kSimplex{
        {4.0, 0.2, 0.2},  {2.0, 0.6, 0.6},  {1.0, 1.0, 1.0},  {0.6, 1.2, 2.5},
        {0.4, 1.5, 4.0},  {0.2, 1.5, 6.0},  {0.05, 1.5, 8.0}, {0.05, 3.0, 2.0},
        {0.3, 3.0, 1.0},  {0.8, 0.3, 4.0},
    };
    const std::size_t num_presets = kSimplex.size();
    std::filesystem::create_directories(args.out_dir);
    std::ofstream(args.out_dir / "presets.json") << nlohmann::json(kSimplex).dump(2) << '\n';

    const student::DbDensityIndex density_idx(db);
    const auto range_idx = student::build_sampled_range_index(db, 128000);
    std::vector<std::vector<std::size_t>> pools(db.size());
    core::parallel_for(db.size(), [&](std::size_t i) {
        pools[i] = student::build_candidate_pool(db.trajectory(i));
    });

    auto eval_profile = [](const core::TrajectoryView &traj, std::span<const std::size_t> kept,
                           std::span<const student::SampledRangeBox> uncov_boxes) {
        double max_ped = 0.0;
        for (std::size_t seg = 0; seg + 1 < kept.size(); ++seg) {
            const std::size_t start_idx = kept[seg], end_idx = kept[seg + 1];
            const std::size_t step = (end_idx - start_idx) > 64 ? ((end_idx - start_idx) / 64) : 1;
            for (std::size_t k = start_idx + 1; k < end_idx; k += step) {
                max_ped = std::max(
                    max_ped, std::sqrt(selection::distance::squared_segment_distance(
                                 traj.x[k], traj.y[k], traj.x[start_idx], traj.y[start_idx],
                                 traj.x[end_idx], traj.y[end_idx])));
            }
        }
        double where_at = 0.0;
        const double start_t = static_cast<double>(traj.t.front());
        const double end_t = static_cast<double>(traj.t.back());
        if (end_t > start_t) {
            std::size_t orig_seg = 0, kept_seg = 0;
            for (int probe = 0; probe < 32; ++probe) {
                const double probe_t = start_t + ((probe + 0.5) / 32.0) * (end_t - start_t);
                while (orig_seg + 2 < traj.size() &&
                       static_cast<double>(traj.t[orig_seg + 1]) < probe_t)
                    ++orig_seg;
                while (kept_seg + 2 < kept.size() &&
                       static_cast<double>(traj.t[kept[kept_seg + 1]]) < probe_t)
                    ++kept_seg;
                auto interp_point = [&](std::size_t idx0, std::size_t idx1) {
                    const double dt = static_cast<double>(traj.t[idx1] - traj.t[idx0]);
                    const double frac =
                        dt > 0.0
                            ? std::clamp((probe_t - static_cast<double>(traj.t[idx0])) / dt, 0.0,
                                         1.0)
                            : 0.0;
                    return std::pair<double, double>{
                        traj.x[idx0] + frac * (traj.x[idx1] - traj.x[idx0]),
                        traj.y[idx0] + frac * (traj.y[idx1] - traj.y[idx0])};
                };
                const auto [orig_x, orig_y] = interp_point(orig_seg, orig_seg + 1);
                const auto [kept_x, kept_y] = interp_point(kept[kept_seg], kept[kept_seg + 1]);
                const double dist = std::hypot(orig_x - kept_x, orig_y - kept_y);
                where_at += dist <= 25.0 ? 1.0 : (dist >= 200.0 ? 0.0 : (200.0 - dist) / 175.0);
            }
            where_at /= 32.0;
        }
        double covered_weight = 0.0, total_weight = 0.0;
        for (const auto &sampled_box : uncov_boxes) {
            total_weight += sampled_box.weight;
            const auto &box = sampled_box.box;
            for (std::size_t idx : kept) {
                const std::int64_t pt_t = traj.t[idx];
                if (pt_t < box.t_min)
                    continue;
                if (pt_t > box.t_max)
                    break;
                if (traj.x[idx] >= box.x_min && traj.x[idx] <= box.x_max &&
                    traj.y[idx] >= box.y_min && traj.y[idx] <= box.y_max) {
                    covered_weight += sampled_box.weight;
                    break;
                }
            }
        }
        return std::array<double, 3>{max_ped, where_at,
                                     total_weight > 1e-12 ? (covered_weight / total_weight) : 1.0};
    };

    auto chamfer = [](const core::TrajectoryView &traj, std::span<const std::size_t> lhs,
                      std::span<const std::size_t> rhs) {
        if (lhs.size() <= 2 || rhs.size() <= 2)
            return 0.0;
        auto one_way = [&](std::span<const std::size_t> src, std::span<const std::size_t> dst) {
            double sum = 0.0;
            for (std::size_t i = 1; i + 1 < src.size(); ++i) {
                double best_dist = 1e18;
                for (std::size_t j = 1; j + 1 < dst.size(); ++j)
                    best_dist =
                        std::min(best_dist, std::hypot(traj.x[src[i]] - traj.x[dst[j]],
                                                       traj.y[src[i]] - traj.y[dst[j]]));
                sum += best_dist;
            }
            return sum / static_cast<double>(src.size() - 2);
        };
        return 0.5 * (one_way(lhs, rhs) + one_way(rhs, lhs));
    };

    for (double share : shares) {
        const auto kept_path = args.out_dir / std::format("teacher_kept_{:g}.bin", share);
        if (!std::filesystem::exists(kept_path))
            continue;
        const auto teacher_kept_all = load_kept_binary(kept_path);
        auto t_start = std::chrono::steady_clock::now();
        std::vector<float> losses(db.size() * num_presets, 0.0f);
        core::parallel_for(db.size(), [&](std::size_t i) {
            const auto traj = db.trajectory(i);
            const auto &teacher_kept = teacher_kept_all[i];
            if (teacher_kept.size() <= 2)
                return;
            const auto target_profile = eval_profile(traj, teacher_kept, range_idx.uncov_boxes[i]);
            for (std::size_t preset_idx = 0; preset_idx < num_presets; ++preset_idx) {
                const student::SubmodularWeights weights{kSimplex[preset_idx][0],
                                                         kSimplex[preset_idx][1],
                                                         kSimplex[preset_idx][2], 1};
                const double total_range_weight =
                    range_idx.uncov_weight[i] + range_idx.ep_weight[i];
                const double uncovered_frac = (total_range_weight > 1e-12)
                                                  ? (range_idx.uncov_weight[i] / total_range_weight)
                                                  : 0.0;
                const auto candidate_kept = student::submodular_simplify_on_pool(
                    traj, pools[i], teacher_kept.size(), weights, density_idx,
                    range_idx.valid_boxes > 0 ? &range_idx.uncov_boxes[i] : nullptr,
                    uncovered_frac);
                const auto candidate_profile =
                    eval_profile(traj, candidate_kept, range_idx.uncov_boxes[i]);
                const double loss =
                    chamfer(traj, candidate_kept, teacher_kept) / 150.0 +
                    std::abs(candidate_profile[0] - target_profile[0]) / 100.0 +
                    std::abs(candidate_profile[1] - target_profile[1]) * 3.0 +
                    std::abs(candidate_profile[2] - target_profile[2]) * 3.0;
                losses[i * num_presets + preset_idx] = static_cast<float>(loss);
            }
        });
        write_raw(args.out_dir / std::format("preset_losses_{:g}.bin", share), losses);
        std::cerr << std::format("fit-simplex share {:g} done in {:.2f}s\n", share,
                                 core::seconds_since(t_start));
    }
}

// =============================================================================
// 3. EVAL: build TestWorkloads once, evaluate Student (+ optional baselines/Teacher)
// =============================================================================
void run_eval(const core::TrajectoryDb &db, const core::TrajectoryDb &pool,
              const pipeline::Config &config, const SplitData &split_data, const CliArgs &args,
              std::uint64_t seed, std::span<const double> shares) {
    std::vector<student::StudentModel> models;
    if (!args.model_paths.empty()) {
        for (const auto &path : args.model_paths)
            models.push_back(student::load_model(path));
    } else if (!config.student.model_path.empty()) {
        models.push_back(student::load_model(config.student.model_path));
    }
    if (models.empty() && !args.baselines && !args.teacher_dir)
        throw std::invalid_argument("eval requires --model, --baselines, or --teacher-dir");

    auto t_workloads = std::chrono::steady_clock::now();
    const pipeline::TestWorkloads test_workloads(db, pool, config);
    const auto test = test_workloads.view();
    const auto floor_db = core::materialize(db, core::endpoints_only(db));
    const auto floor_scores =
        evaluator::Evaluator(db, floor_db, test, config.where_at_scoring).evaluate().scores;
    std::cerr << std::format("[eval] TestWorkloads+floor built in {:.2f}s\n",
                             core::seconds_since(t_workloads));

    std::optional<student::DbDensityIndex> density_idx;
    std::vector<std::vector<std::size_t>> pools;
    if (args.baselines) {
        density_idx.emplace(db);
        pools.resize(db.size());
        core::parallel_for(db.size(), [&](std::size_t i) {
            pools[i] = student::build_candidate_pool(db.trajectory(i));
        });
    }

    std::vector<query::JoinWorkload> unseen_joins;
    std::vector<double> unseen_floors;
    for (double eps : {75.0, 125.0, 175.0}) {
        unseen_joins.push_back(query::generate_join_queries(
            db, pool,
            {.query_count = config.join.query_count,
             .eps_m = eps,
             .mode = query::JoinMode::HeldOut,
             .frechet = config.join.frechet},
            seed));
        unseen_floors.push_back(evaluator::evaluate_join(floor_db, unseen_joins.back()).f1);
    }

    std::optional<std::ofstream> csv_out;
    if (args.csv_path) {
        if (args.csv_path->has_parent_path())
            std::filesystem::create_directories(args.csv_path->parent_path());
        const bool write_header = !std::filesystem::exists(*args.csv_path);
        csv_out.emplace(*args.csv_path, std::ios::app);
        if (write_header)
            *csv_out << "split,seed,method,share_pct,j4,knn,range,where_at,join,ujoin,dec_ms\n";
    }

    auto report = [&](const std::string &label, double share, std::size_t total_budget,
                      const core::KeptSets &kept, double dec_ms) {
        validate_kept(kept, db, total_budget);
        const auto simplified_db = core::materialize(db, kept);
        const auto raw_scores =
            evaluator::Evaluator(db, simplified_db, test, config.where_at_scoring)
                .evaluate()
                .scores;
        const auto norm_scores = evaluator::normalize_scores(raw_scores, floor_scores);

        double unseen_sum = 0.0;
        for (std::size_t idx = 0; idx < unseen_joins.size(); ++idx) {
            const double f1 = evaluator::evaluate_join(simplified_db, unseen_joins[idx]).f1;
            unseen_sum += (f1 - unseen_floors[idx]) / (1.0 - unseen_floors[idx]);
        }
        const double unseen_join = unseen_sum / 3.0;

        std::cout << std::format(
            "RESULT | {:<20} | {:5.2f}% | J4 {:6.4f} | kNN {:6.4f} | Rng {:6.4f} | WA {:6.4f} | "
            "Join {:6.4f} | uJoin {:6.4f} | dec {:6.1f}ms\n",
            label, share * 100.0, norm_scores.combined, norm_scores.knn, norm_scores.range,
            norm_scores.where_at, norm_scores.join, unseen_join, dec_ms);
        std::cout.flush();
        if (csv_out) {
            *csv_out << std::format(
                "{},{},{},{:.2f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.1f}\n", args.split,
                seed, label, share * 100.0, norm_scores.combined, norm_scores.knn,
                norm_scores.range, norm_scores.where_at, norm_scores.join, unseen_join, dec_ms);
            csv_out->flush();
        }
    };

    auto run_student = [&](const std::string &label, double share, std::size_t total_budget,
                           bool rdp_only) {
        const auto result =
            (models.size() > 1)
                ? student::simplify_database_oof(db, total_budget, models,
                                                 split_data.fold_assignment,
                                                 config.student.rb_passes, rdp_only)
                : student::simplify_database(db, total_budget, models[0], config.student.rb_passes,
                                             rdp_only);
        const double total_ms = result.density_ms + result.feature_ms + result.ml_ms +
                                result.pool_ms + result.decode_ms;
        std::cerr << std::format(
            "[pure-C++ {} {:5.2f}%] dens={:.1f}ms feat={:.1f}ms ml={:.1f}ms pool={:.1f}ms "
            "dec={:.1f}ms -> total={:.1f}ms\n",
            label, share * 100.0, result.density_ms, result.feature_ms, result.ml_ms,
            result.pool_ms, result.decode_ms, total_ms);
        report(label, share, total_budget, result.kept, total_ms);
    };

    for (double share : shares) {
        const std::size_t total_budget = allocation::total_budget(db, share);
        if (args.baselines) {
            const auto equal_budgets = allocation::create_trajectory_budgets(
                db, allocation::AllocatorType::SizeProportional, total_budget, 0.0);
            core::KeptSets kept(db.size());
            auto t_start = std::chrono::steady_clock::now();
            core::parallel_for(db.size(), [&](std::size_t i) {
                kept[i] = selection::rdp::simplify(db.trajectory(i), equal_budgets[i]);
            });
            report("RDP_a0", share, total_budget, kept, core::seconds_since(t_start) * 1000.0);

            t_start = std::chrono::steady_clock::now();
            core::parallel_for(db.size(), [&](std::size_t i) {
                kept[i] = selection::sed::simplify(db.trajectory(i), equal_budgets[i]);
            });
            report("SED_a0", share, total_budget, kept, core::seconds_since(t_start) * 1000.0);

            t_start = std::chrono::steady_clock::now();
            const student::SubmodularWeights weights{0.40, 1.30, 2.25, config.student.rb_passes};
            core::parallel_for(db.size(), [&](std::size_t i) {
                kept[i] = student::submodular_simplify_on_pool(
                    db.trajectory(i), pools[i], equal_budgets[i], weights, *density_idx);
            });
            report("ZeroML_SubmodV3", share, total_budget, kept,
                   core::seconds_since(t_start) * 1000.0);

            if (!models.empty())
                run_student("S1_DWF+RDP", share, total_budget, true);
        }
        if (!models.empty())
            run_student("Student", share, total_budget, false);
        if (args.teacher_dir) {
            const auto rdp_path =
                *args.teacher_dir / std::format("teacher_rdp_budgets_{:g}.bin", share);
            if (std::filesystem::exists(rdp_path)) {
                const auto rdp_budgets = read_raw<std::uint32_t>(rdp_path);
                core::KeptSets kept(db.size());
                auto t_start = std::chrono::steady_clock::now();
                core::parallel_for(db.size(), [&](std::size_t i) {
                    kept[i] = selection::rdp::simplify(db.trajectory(i), rdp_budgets[i]);
                });
                report("TEACHER_BUDGET_RDP", share, total_budget, kept,
                       core::seconds_since(t_start) * 1000.0);
            }
            const auto refine_path =
                *args.teacher_dir / std::format("teacher_kept_{:g}.bin", share);
            if (std::filesystem::exists(refine_path))
                report("TEACHER_REFINE", share, total_budget, load_kept_binary(refine_path), 0.0);
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: student_tool <export|fit-simplex|eval> [options]\n";
        return 1;
    }
    const auto args = parse_args(argc, argv);

    auto config = pipeline::load_config(args.config_path);
    const std::uint64_t seed = args.seed_override.value_or(config.seed);
    config.seed = seed;
    config.self_join = false;
    config.clustering.enabled = false;

    const auto split_data = load_split(config, args, seed);
    std::cerr << std::format("[student_tool {}] split={} seed={} DB={} trajs ({} pts), Q={} trajs\n",
                             args.cmd, args.split, seed, split_data.db.size(),
                             split_data.db.num_points(), split_data.pool.size());

    const auto &export_shares =
        (args.split == "full") ? config.budget_shares : config.student.train_shares;

    if (args.cmd == "export") {
        run_export(split_data.db, config, split_data, args, seed, export_shares);
        return 0;
    }
    if (args.cmd == "fit-simplex") {
        run_fit_simplex(split_data.db, args, config.student.train_shares);
        return 0;
    }
    if (args.cmd == "eval") {
        run_eval(split_data.db, split_data.pool, config, split_data, args, seed,
                 config.budget_shares);
        return 0;
    }

    std::cerr << "unknown command: " << args.cmd << "\n";
    return 1;
}

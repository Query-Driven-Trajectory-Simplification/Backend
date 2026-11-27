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
void write_raw(const std::filesystem::path &path, const std::vector<T> &vec) {
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot open for write: " + path.string());
    if (!vec.empty())
        out.write(reinterpret_cast<const char *>(vec.data()),
                  static_cast<std::streamsize>(vec.size() * sizeof(T)));
}

template <typename T> std::vector<T> read_raw(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("cannot open for read: " + path.string());
    const auto sz = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    std::vector<T> vec(sz / sizeof(T));
    if (!vec.empty())
        in.read(reinterpret_cast<char *>(vec.data()), static_cast<std::streamsize>(sz));
    return vec;
}

void save_kept_binary(const std::filesystem::path &path, const core::KeptSets &kept) {
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t n = kept.size();
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const auto &k : kept) {
        const auto m = static_cast<std::uint32_t>(k.size());
        out.write(reinterpret_cast<const char *>(&m), sizeof(m));
        for (std::size_t idx : k) {
            const auto v = static_cast<std::uint32_t>(idx);
            out.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
    }
}

core::KeptSets load_kept_binary(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open kept binary: " + path.string());
    std::uint64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    core::KeptSets kept(n);
    for (std::uint64_t i = 0; i < n; ++i) {
        std::uint32_t m = 0;
        in.read(reinterpret_cast<char *>(&m), sizeof(m));
        kept[i].resize(m);
        for (std::uint32_t j = 0; j < m; ++j) {
            std::uint32_t v = 0;
            in.read(reinterpret_cast<char *>(&v), sizeof(v));
            kept[i][j] = v;
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
    CliArgs a;
    a.cmd = argv[1];
    for (int k = 2; k < argc; ++k) {
        const std::string key = argv[k];
        auto val = [&] {
            if (k + 1 >= argc)
                throw std::invalid_argument("missing value for " + key);
            return std::string(argv[++k]);
        };
        if (key == "--config") a.config_path = val();
        else if (key == "--split") a.split = val();
        else if (key == "--seed") a.seed_override = std::stoull(val());
        else if (key == "--pseudo-split") a.pseudo_split = std::stoull(val());
        else if (key == "--out-dir") a.out_dir = val();
        else if (key == "--baselines") a.baselines = true;
        else if (key == "--model") {
            std::stringstream ss(val());
            for (std::string item; std::getline(ss, item, ',');)
                a.model_paths.emplace_back(item);
        } else if (key == "--teacher-dir") a.teacher_dir = val();
        else if (key == "--csv") a.csv_path = val();
        else throw std::invalid_argument("unknown flag: " + key);
    }
    return a;
}

struct SplitData {
    core::TrajectoryDb db;
    core::TrajectoryDb pool;
    std::filesystem::path cache_dir;
    std::vector<std::uint32_t> fold_assignment;
};

SplitData load_split(const pipeline::Config &config, const CliArgs &args, std::uint64_t seed) {
    const auto all = io::load_trajectories(config.data_path);
    const auto outer = split::split(all, config.split, seed);

    core::TrajectoryDb base_db = core::subset(all, outer.database);
    core::TrajectoryDb base_pool = core::subset(all, outer.queries);
    std::filesystem::path cache_dir = config.teacher.cache_dir;

    if (args.pseudo_split) {
        const auto parts = split::split(base_db, config.split, *args.pseudo_split);
        auto inner_db = core::subset(base_db, parts.database);
        auto inner_pool = core::subset(base_db, parts.queries);
        base_db = std::move(inner_db);
        base_pool = std::move(inner_pool);
        cache_dir = config.teacher.cache_dir / std::format("pseudo_{}", *args.pseudo_split);
    }

    std::vector<std::size_t> perm(base_db.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937_64 rng(seed * 1315423911ULL + (args.pseudo_split.value_or(0) * 2654435761ULL) + 17);
    std::shuffle(perm.begin(), perm.end(), rng);

    const std::size_t mid = perm.size() / 2;
    std::vector<std::uint32_t> fold_of(base_db.size(), 0);
    for (std::size_t pos = 0; pos < perm.size(); ++pos)
        fold_of[perm[pos]] = (pos < mid) ? 0U : 1U;

    if (args.split == "half0" || args.split == "half1") {
        const int half = (args.split == "half1") ? 1 : 0;
        std::vector<std::size_t> idx =
            (half == 0) ? std::vector<std::size_t>(perm.begin(), perm.begin() + static_cast<std::ptrdiff_t>(mid))
                        : std::vector<std::size_t>(perm.begin() + static_cast<std::ptrdiff_t>(mid), perm.end());
        std::ranges::sort(idx);
        return {core::subset(base_db, idx), std::move(base_pool),
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
        const auto n = db.trajectory(i).size();
        const auto &k = kept[i];
        if (k.size() < std::min<std::size_t>(2, n) || k.size() > n)
            throw std::runtime_error(std::format("traj {} invalid budget {} (n={})", i, k.size(), n));
        if (k.front() != 0 || k.back() != n - 1)
            throw std::runtime_error(std::format("traj {} missing endpoints", i));
        for (std::size_t j = 1; j < k.size(); ++j)
            if (k[j] <= k[j - 1])
                throw std::runtime_error(std::format("traj {} non-increasing indices", i));
        total += k.size();
    }
    if (total != expected_budget)
        throw std::runtime_error(std::format("budget mismatch: got {} expected {}", total, expected_budget));
}

// =============================================================================
// 1. EXPORT: tiny10 features + cached frozen Teacher targets
// =============================================================================
void run_export(const core::TrajectoryDb &db, const pipeline::Config &config,
                const SplitData &sp, const CliArgs &args, std::uint64_t seed,
                std::span<const double> shares) {
    std::filesystem::create_directories(args.out_dir);

    auto t0 = std::chrono::steady_clock::now();
    const student::DbDensityIndex density_idx(db);
    const double dens_ms = core::seconds_since(t0) * 1000.0;

    auto t1 = std::chrono::steady_clock::now();
    const auto feats = student::extract_trajectory_features(db, density_idx);
    const double feat_ms = core::seconds_since(t1) * 1000.0;

    std::vector<std::uint32_t> lengths(db.size());
    for (std::size_t i = 0; i < db.size(); ++i)
        lengths[i] = static_cast<std::uint32_t>(db.trajectory(i).size());

    write_raw(args.out_dir / "features.bin", feats.values);
    write_raw(args.out_dir / "lengths.bin", lengths);

    nlohmann::json meta{
        {"num_trajs", feats.num_trajs},
        {"num_points", db.num_points()},
        {"num_features", feats.feature_names.size()},
        {"feature_names", feats.feature_names},
        {"shares", std::vector<double>(shares.begin(), shares.end())},
        {"density_ms", dens_ms},
        {"feature_ms", feat_ms},
    };
    std::ofstream(args.out_dir / "meta.json") << meta.dump(2) << '\n';
    std::cerr << std::format("Exported {} trajs x {} feats in {:.1f}ms dens + {:.1f}ms feat\n",
                             feats.num_trajs, feats.feature_names.size(), dens_ms, feat_ms);

    bool any_missing = false;
    for (double s : shares)
        if (!std::filesystem::exists(args.out_dir / std::format("teacher_kept_{:g}.bin", s)))
            any_missing = true;
    if (!any_missing)
        return;

    std::filesystem::create_directories(sp.cache_dir);
    auto tw0 = std::chrono::steady_clock::now();
    auto workloads = teacher::load_or_build_workloads(
        db, pipeline::teacher_workload_config(config), seed, sp.cache_dir);
    if (config.teacher.term_weights[teacher::Objective::Margin] > 0)
        teacher::add_ranked_neighbours(db, workloads, teacher::kMarginOutside, seed, sp.cache_dir);
    const teacher::Teacher model(db, std::move(workloads),
                                 pipeline::teacher_objective_config(config), seed);
    std::cerr << std::format("Teacher setup in {:.2f}s\n", core::seconds_since(tw0));

    const std::vector<std::string> arms{"BUDGET_RDP", "BUDGET_LOOP_REFINE"};
    for (double s : shares) {
        const auto kept_path = args.out_dir / std::format("teacher_kept_{:g}.bin", s);
        if (std::filesystem::exists(kept_path))
            continue;
        auto ts = std::chrono::steady_clock::now();
        auto res = model.run(allocation::total_budget(db, s), arms);
        auto rdp_kept = core::to_kept(res[0].simplification);
        auto ref_kept = core::to_kept(res[1].simplification);

        std::vector<std::uint32_t> b_rdp(db.size()), b_ref(db.size());
        for (std::size_t i = 0; i < db.size(); ++i) {
            b_rdp[i] = static_cast<std::uint32_t>(rdp_kept[i].size());
            b_ref[i] = static_cast<std::uint32_t>(ref_kept[i].size());
        }
        write_raw(args.out_dir / std::format("teacher_rdp_budgets_{:g}.bin", s), b_rdp);
        write_raw(args.out_dir / std::format("teacher_budgets_{:g}.bin", s), b_ref);
        save_kept_binary(kept_path, ref_kept);
        std::cerr << std::format("  share {:g} Teacher saved in {:.2f}s\n", s,
                                 core::seconds_since(ts));
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
    const std::size_t G = kSimplex.size();
    std::filesystem::create_directories(args.out_dir);
    std::ofstream(args.out_dir / "presets.json") << nlohmann::json(kSimplex).dump(2) << '\n';

    const student::DbDensityIndex density_idx(db);
    std::vector<std::vector<float>> all_rho(db.size());
    std::vector<std::vector<std::size_t>> pools(db.size());
    core::parallel_for(db.size(), [&](std::size_t i) {
        all_rho[i] = density_idx.trajectory_range_rho(db.trajectory(i));
        pools[i] = student::build_candidate_pool(db.trajectory(i));
    });

    auto eval_profile = [](const core::TrajectoryView &tr, std::span<const std::size_t> kept,
                           std::span<const float> rho) {
        double max_ped = 0.0;
        for (std::size_t s = 0; s + 1 < kept.size(); ++s) {
            const std::size_t a = kept[s], b = kept[s + 1];
            const std::size_t step = (b - a) > 64 ? ((b - a) / 64) : 1;
            for (std::size_t k = a + 1; k < b; k += step) {
                max_ped = std::max(max_ped, std::sqrt(selection::distance::squared_segment_distance(
                                                tr.x[k], tr.y[k], tr.x[a], tr.y[a], tr.x[b], tr.y[b])));
            }
        }
        double wa = 0.0;
        const double t0 = static_cast<double>(tr.t.front()), t1 = static_cast<double>(tr.t.back());
        if (t1 > t0) {
            std::size_t os = 0, ks = 0;
            for (int p = 0; p < 32; ++p) {
                const double tp = t0 + ((p + 0.5) / 32.0) * (t1 - t0);
                while (os + 2 < tr.size() && static_cast<double>(tr.t[os + 1]) < tp) ++os;
                while (ks + 2 < kept.size() && static_cast<double>(tr.t[kept[ks + 1]]) < tp) ++ks;
                auto ip = [&](std::size_t i0, std::size_t i1) {
                    const double dt = static_cast<double>(tr.t[i1] - tr.t[i0]);
                    const double u = dt > 0.0 ? std::clamp((tp - static_cast<double>(tr.t[i0])) / dt, 0.0, 1.0) : 0.0;
                    return std::pair<double, double>{tr.x[i0] + u * (tr.x[i1] - tr.x[i0]),
                                                     tr.y[i0] + u * (tr.y[i1] - tr.y[i0])};
                };
                const auto [ox, oy] = ip(os, os + 1);
                const auto [kx, ky] = ip(kept[ks], kept[ks + 1]);
                const double d = std::hypot(ox - kx, oy - ky);
                wa += d <= 25.0 ? 1.0 : (d >= 200.0 ? 0.0 : (200.0 - d) / 175.0);
            }
            wa /= 32.0;
        }
        std::vector<std::pair<std::int64_t, float>> cells;
        cells.reserve(kept.size());
        for (std::size_t idx : kept) {
            const auto gx = static_cast<std::int64_t>(std::floor(tr.x[idx] / 500.0));
            const auto gy = static_cast<std::int64_t>(std::floor(tr.y[idx] / 500.0));
            cells.emplace_back((gx << 32) ^ (gy & 0xffffffffLL), rho[idx]);
        }
        std::ranges::sort(cells, [](auto a, auto b) {
            return a.first < b.first || (a.first == b.first && a.second > b.second);
        });
        double cov = 0.0;
        for (std::size_t j = 0; j < cells.size(); ++j)
            if (j == 0 || cells[j].first != cells[j - 1].first)
                cov += cells[j].second;
        return std::array<double, 3>{max_ped, wa, cov};
    };

    auto chamfer = [](const core::TrajectoryView &tr, std::span<const std::size_t> a,
                      std::span<const std::size_t> b) {
        if (a.size() <= 2 || b.size() <= 2)
            return 0.0;
        auto one_way = [&](std::span<const std::size_t> u, std::span<const std::size_t> v) {
            double sum = 0.0;
            for (std::size_t i = 1; i + 1 < u.size(); ++i) {
                double best = 1e18;
                for (std::size_t j = 1; j + 1 < v.size(); ++j)
                    best = std::min(best, std::hypot(tr.x[u[i]] - tr.x[v[j]],
                                                     tr.y[u[i]] - tr.y[v[j]]));
                sum += best;
            }
            return sum / static_cast<double>(u.size() - 2);
        };
        return 0.5 * (one_way(a, b) + one_way(b, a));
    };

    for (double s : shares) {
        const auto kept_path = args.out_dir / std::format("teacher_kept_{:g}.bin", s);
        if (!std::filesystem::exists(kept_path))
            continue;
        const auto t_kept = load_kept_binary(kept_path);
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> losses(db.size() * G, 0.0f);
        core::parallel_for(db.size(), [&](std::size_t i) {
            const auto tr = db.trajectory(i);
            const auto &tk = t_kept[i];
            if (tk.size() <= 2)
                return;
            const auto tgt = eval_profile(tr, tk, all_rho[i]);
            for (std::size_t g = 0; g < G; ++g) {
                const student::SubmodularWeights sw{kSimplex[g][0], kSimplex[g][1], kSimplex[g][2], 1};
                const auto ck = student::submodular_simplify_on_pool(tr, pools[i], tk.size(),
                                                                     sw, density_idx);
                const auto prof = eval_profile(tr, ck, all_rho[i]);
                const double loss = chamfer(tr, ck, tk) / 150.0 +
                                    std::abs(prof[0] - tgt[0]) / 100.0 +
                                    std::abs(prof[1] - tgt[1]) * 3.0 +
                                    std::abs(prof[2] - tgt[2]) / std::max(1.0, tgt[2]) * 2.0;
                losses[i * G + g] = static_cast<float>(loss);
            }
        });
        write_raw(args.out_dir / std::format("preset_losses_{:g}.bin", s), losses);
        std::cerr << std::format("fit-simplex share {:g} done in {:.2f}s\n", s,
                                 core::seconds_since(t0));
    }
}

// =============================================================================
// 3. EVAL: build TestWorkloads once, evaluate Student (+ optional baselines/Teacher)
// =============================================================================
void run_eval(const core::TrajectoryDb &db, const core::TrajectoryDb &pool,
              const pipeline::Config &config, const SplitData &sp, const CliArgs &args,
              std::uint64_t seed, std::span<const double> shares) {
    std::vector<student::StudentModel> models;
    if (!args.model_paths.empty()) {
        for (const auto &p : args.model_paths)
            models.push_back(student::load_model(p));
    } else if (!config.student.model_path.empty()) {
        models.push_back(student::load_model(config.student.model_path));
    }
    if (models.empty() && !args.baselines && !args.teacher_dir)
        throw std::invalid_argument("eval requires --model, --baselines, or --teacher-dir");

    auto tw0 = std::chrono::steady_clock::now();
    const pipeline::TestWorkloads test_workloads(db, pool, config);
    const auto test = test_workloads.view();
    const auto floor_db = core::materialize(db, core::endpoints_only(db));
    const auto floor =
        evaluator::Evaluator(db, floor_db, test, config.where_at_scoring).evaluate().scores;
    std::cerr << std::format("[eval] TestWorkloads+floor built in {:.2f}s\n",
                             core::seconds_since(tw0));

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
        const bool write_hdr = !std::filesystem::exists(*args.csv_path);
        csv_out.emplace(*args.csv_path, std::ios::app);
        if (write_hdr)
            *csv_out << "split,seed,method,share_pct,j4,knn,range,where_at,join,ujoin,dec_ms\n";
    }

    auto report = [&](const std::string &label, double share, std::size_t total_budget,
                      const core::KeptSets &kept, double dec_ms) {
        validate_kept(kept, db, total_budget);
        const auto comp = core::materialize(db, kept);
        const auto scores =
            evaluator::Evaluator(db, comp, test, config.where_at_scoring).evaluate().scores;
        const auto ns = evaluator::normalize_scores(scores, floor);

        double usum = 0.0;
        for (std::size_t u = 0; u < unseen_joins.size(); ++u) {
            const double v = evaluator::evaluate_join(comp, unseen_joins[u]).f1;
            usum += (v - unseen_floors[u]) / (1.0 - unseen_floors[u]);
        }
        const double uj = usum / 3.0;

        std::cout << std::format(
            "RESULT | {:<20} | {:5.2f}% | J4 {:6.4f} | kNN {:6.4f} | Rng {:6.4f} | WA {:6.4f} | "
            "Join {:6.4f} | uJoin {:6.4f} | dec {:6.1f}ms\n",
            label, share * 100.0, ns.combined, ns.knn, ns.range, ns.where_at, ns.join, uj, dec_ms);
        std::cout.flush();
        if (csv_out) {
            *csv_out << std::format("{},{},{},{:.2f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.1f}\n",
                                    args.split, seed, label, share * 100.0, ns.combined,
                                    ns.knn, ns.range, ns.where_at, ns.join, uj, dec_ms);
            csv_out->flush();
        }
    };

    auto run_student = [&](const std::string &label, double share, std::size_t total_budget,
                           bool rdp_only) {
        const auto res = (models.size() > 1)
                             ? student::simplify_database_oof(db, total_budget, models,
                                                              sp.fold_assignment,
                                                              config.student.rb_passes, rdp_only)
                             : student::simplify_database(db, total_budget, models[0],
                                                          config.student.rb_passes, rdp_only);
        const double ms =
            res.density_ms + res.feature_ms + res.ml_ms + res.pool_ms + res.decode_ms;
        std::cerr << std::format(
            "[pure-C++ {} {:5.2f}%] dens={:.1f}ms feat={:.1f}ms ml={:.1f}ms pool={:.1f}ms dec={:.1f}ms -> total={:.1f}ms\n",
            label, share * 100.0, res.density_ms, res.feature_ms, res.ml_ms, res.pool_ms,
            res.decode_ms, ms);
        report(label, share, total_budget, res.kept, ms);
    };

    for (double s : shares) {
        const std::size_t B = allocation::total_budget(db, s);
        if (args.baselines) {
            const auto b_a0 = allocation::create_trajectory_budgets(
                db, allocation::AllocatorType::SizeProportional, B, 0.0);
            core::KeptSets kept(db.size());
            auto t0 = std::chrono::steady_clock::now();
            core::parallel_for(db.size(), [&](std::size_t i) {
                kept[i] = selection::rdp::simplify(db.trajectory(i), b_a0[i]);
            });
            report("RDP_a0", s, B, kept, core::seconds_since(t0) * 1000.0);

            t0 = std::chrono::steady_clock::now();
            core::parallel_for(db.size(), [&](std::size_t i) {
                kept[i] = selection::sed::simplify(db.trajectory(i), b_a0[i]);
            });
            report("SED_a0", s, B, kept, core::seconds_since(t0) * 1000.0);

            t0 = std::chrono::steady_clock::now();
            const student::SubmodularWeights sw{0.40, 1.30, 2.25, config.student.rb_passes};
            core::parallel_for(db.size(), [&](std::size_t i) {
                kept[i] = student::submodular_simplify_on_pool(db.trajectory(i), pools[i],
                                                               b_a0[i], sw, *density_idx);
            });
            report("ZeroML_SubmodV3", s, B, kept, core::seconds_since(t0) * 1000.0);

            if (!models.empty())
                run_student("S1_DWF+RDP", s, B, true);
        }
        if (!models.empty())
            run_student("Student", s, B, false);
        if (args.teacher_dir) {
            const auto rdp_p = *args.teacher_dir / std::format("teacher_rdp_budgets_{:g}.bin", s);
            if (std::filesystem::exists(rdp_p)) {
                const auto raw_b = read_raw<std::uint32_t>(rdp_p);
                core::KeptSets kept(db.size());
                auto t0 = std::chrono::steady_clock::now();
                core::parallel_for(db.size(), [&](std::size_t i) {
                    kept[i] = selection::rdp::simplify(db.trajectory(i), raw_b[i]);
                });
                report("TEACHER_BUDGET_RDP", s, B, kept, core::seconds_since(t0) * 1000.0);
            }
            const auto ref_p = *args.teacher_dir / std::format("teacher_kept_{:g}.bin", s);
            if (std::filesystem::exists(ref_p))
                report("TEACHER_REFINE", s, B, load_kept_binary(ref_p), 0.0);
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

    const auto sp = load_split(config, args, seed);
    std::cerr << std::format("[student_tool {}] split={} seed={} DB={} trajs ({} pts), Q={} trajs\n",
                             args.cmd, args.split, seed, sp.db.size(), sp.db.num_points(),
                             sp.pool.size());

    const auto &export_shares =
        (args.split == "full") ? config.budget_shares : config.student.train_shares;

    if (args.cmd == "export") {
        run_export(sp.db, config, sp, args, seed, export_shares);
        return 0;
    }
    if (args.cmd == "fit-simplex") {
        run_fit_simplex(sp.db, args, config.student.train_shares);
        return 0;
    }
    if (args.cmd == "eval") {
        run_eval(sp.db, sp.pool, config, sp, args, seed, config.budget_shares);
        return 0;
    }

    std::cerr << "unknown command: " << args.cmd << "\n";
    return 1;
}

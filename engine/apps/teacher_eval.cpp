// Runs the teacher and baselines on several seeds and scores every arm on the held-out test
// workloads, built only after every arm is done. Settings come from the --config JSON; flags
// override parts of it. The database self join is the teacher's training data, so it is reported
// but kept out of J4 when join.mode is HeldOut. Writes CSVs to --out; summarise with
// teacher_stats.py.
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/timer.hpp"
#include "backend/engine/evaluator/combined.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/join.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/io/selection_file.hpp"
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/pipeline/test_workloads.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/route_clustering.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/split/split.hpp"
#include "backend/engine/teacher/teacher.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using core::KeptSets;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr std::array kEvalEps{100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0};
constexpr double kRdpAlpha = 0.25; // RDP_a025
constexpr std::array kPartnerLambdas{0.5, 1.0};
constexpr std::size_t kExportBoxes = 100;
constexpr std::uint64_t kExportBoxSeedOffset = 7101;
// Unset options keep the config's value; arms default to all teacher arms
struct Args {
    std::filesystem::path config = "engine/configs/geolife.json";
    std::filesystem::path out = "results/teacher";
    std::optional<std::filesystem::path> data;
    std::optional<std::filesystem::path> cache;
    std::optional<std::vector<std::uint64_t>> seeds; // config: seed
    std::optional<std::vector<double>> shares;       // config: budget_shares
    std::vector<std::string> arms{teacher::kArms.begin(), teacher::kArms.end()};
    bool baselines = true;
    std::optional<bool> all_join_pairs;
    std::optional<query::JoinMode> teacher_join;     // config: teacher.join, else join.mode
    std::optional<std::vector<double>> train_eps;    // config: teacher.extra_join_eps_m
    std::optional<bool> held_out_term;               // config: teacher.held_out_term
    std::optional<bool> ranking_knn;                 // config: teacher.ranking_knn
    std::optional<bool> compressed_knn_query;        // config: teacher.knn_query
    std::optional<std::vector<double>> term_weights; // config: teacher.term_weights
    std::vector<double> eval_eps{kEvalEps.begin(), kEvalEps.end()};
    bool knn_check = false;
    std::optional<std::filesystem::path> export_db;
    std::vector<std::string> external;
    std::optional<std::filesystem::path> external_dir;
};

template <typename T> std::vector<T> parse_list(const std::string &text) {
    std::vector<T> items;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, ',');) {
        if constexpr (std::is_same_v<T, std::string>)
            items.push_back(item);
        else
            items.push_back(static_cast<T>(std::stod(item)));
    }
    return items;
}

Args parse_args(int argc, char **argv) {
    Args args;
    for (int k = 1; k < argc; ++k) {
        const std::string key = argv[k];
        auto value = [&] {
            if (k + 1 >= argc)
                throw std::invalid_argument("missing value for " + key);
            return std::string(argv[++k]);
        };
        if (key == "--config")
            args.config = value();
        else if (key == "--data")
            args.data = value();
        else if (key == "--out")
            args.out = value();
        else if (key == "--cache")
            args.cache = value();
        else if (key == "--seeds")
            args.seeds = parse_list<std::uint64_t>(value());
        else if (key == "--shares")
            args.shares = parse_list<double>(value());
        else if (key == "--arms")
            args.arms = parse_list<std::string>(value());
        else if (key == "--train-eps")
            args.train_eps = parse_list<double>(value());
        else if (key == "--eval-eps")
            args.eval_eps = parse_list<double>(value());
        else if (key == "--term-weights") {
            args.term_weights = parse_list<double>(value());
            if (args.term_weights->size() != teacher::kTermCount)
                throw std::invalid_argument("--term-weights takes kNN,range,where-at,join");
        } else if (key == "--independent-knn")
            args.ranking_knn = false;
        else if (key == "--held-out-term")
            args.held_out_term = true;
        else if (key == "--knn-query") {
            const auto query = value();
            if (query != "Original" && query != "Compressed")
                throw std::invalid_argument("--knn-query must be Original or Compressed");
            args.compressed_knn_query = query == "Compressed";
        } else if (key == "--knn-check")
            args.knn_check = true;
        else if (key == "--teacher-join") {
            const auto mode = value();
            if (mode != "HeldOut" && mode != "Self")
                throw std::invalid_argument("--teacher-join must be HeldOut or Self");
            args.teacher_join = mode == "Self" ? query::JoinMode::Self : query::JoinMode::HeldOut;
        } else if (key == "--no-baselines")
            args.baselines = false;
        else if (key == "--seen-pairs-only")
            args.all_join_pairs = false;
        else if (key == "--no-teacher")
            args.arms.clear();
        else if (key == "--export-db")
            args.export_db = value();
        else if (key == "--external")
            args.external = parse_list<std::string>(value());
        else if (key == "--external-dir")
            args.external_dir = value();
        else
            throw std::invalid_argument("unknown argument " + key +
                                        " (--config --data --out --cache --seeds --shares --arms "
                                        "--teacher-join --train-eps --held-out-term "
                                        "--independent-knn --knn-query --term-weights --eval-eps "
                                        "--knn-check --no-baselines --seen-pairs-only --no-teacher "
                                        "--export-db --external --external-dir)");
    }
    teacher::check_arms(args.arms);
    if (!args.external.empty() && !args.external_dir)
        throw std::invalid_argument("--external needs --external-dir");
    return args;
}

pipeline::Config effective_config(const Args &args) {
    auto config = pipeline::load_config(args.config);
    if (args.data)
        config.data_path = *args.data;
    if (args.cache)
        config.teacher.cache_dir = *args.cache;
    if (args.shares)
        config.budget_shares = *args.shares;
    if (args.all_join_pairs)
        config.teacher.all_join_pairs = *args.all_join_pairs;
    if (args.teacher_join)
        config.teacher.join = *args.teacher_join;
    if (args.train_eps)
        config.teacher.extra_join_eps_m = *args.train_eps;
    if (args.held_out_term)
        config.teacher.held_out_term = *args.held_out_term;
    if (args.ranking_knn)
        config.teacher.ranking_knn = *args.ranking_knn;
    if (args.compressed_knn_query)
        config.teacher.compressed_knn_query = *args.compressed_knn_query;
    if (args.term_weights)
        std::ranges::copy(*args.term_weights, config.teacher.term_weights.begin());
    config.self_join = true;
    return config;
}

const char *mode_name(query::JoinMode mode) {
    return mode == query::JoinMode::Self ? "Self" : "HeldOut";
}

// Interior points shared in proportion to w, capped at each trajectory's interior, the rest by
// largest remainder. Budgets include the endpoints
std::vector<std::size_t> weighted_budgets(const core::TrajectoryDb &db,
                                          const std::vector<double> &weights,
                                          std::size_t interior_total) {
    const std::size_t n = db.size();
    std::vector<std::size_t> cap(n), interior(n, 0);
    for (std::size_t i = 0; i < n; ++i)
        cap[i] = db.trajectory(i).size() >= 2 ? db.trajectory(i).size() - 2 : 0;
    std::vector<char> open(n, 1);
    std::size_t left = interior_total;
    while (left > 0) {
        double weight_sum = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (open[i])
                weight_sum += weights[i];
        if (weight_sum <= 0)
            break;
        std::vector<double> frac(n, 0);
        std::size_t given = 0;
        bool capped = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (!open[i])
                continue;
            const double want = static_cast<double>(left) * weights[i] / weight_sum;
            auto add = static_cast<std::size_t>(want);
            if (interior[i] + add >= cap[i]) {
                add = cap[i] - interior[i];
                open[i] = 0;
                capped = true;
            } else {
                frac[i] = want - static_cast<double>(add);
            }
            interior[i] += add;
            given += add;
        }
        left -= given;
        if (capped)
            continue;
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < n; ++i)
            if (open[i])
                idx.push_back(i);
        std::ranges::sort(idx, [&](auto a, auto b) {
            return frac[a] > frac[b] || (frac[a] == frac[b] && a < b);
        });
        for (std::size_t k = 0; k < idx.size() && left > 0; ++k) {
            ++interior[idx[k]];
            --left;
            if (interior[idx[k]] == cap[idx[k]])
                open[idx[k]] = 0;
        }
        if (given == 0 && idx.empty())
            break;
    }
    for (std::size_t i = 0; i < n; ++i)
        interior[i] += std::min<std::size_t>(2, db.trajectory(i).size());
    return interior;
}

KeptSets select(const core::TrajectoryDb &db, selection::AlgorithmType algorithm,
                const std::vector<std::size_t> &budgets) {
    KeptSets kept(db.size());
    core::parallel_for(db.size(), [&](std::size_t i) {
        kept[i] = selection::simplify_trajectory(db.trajectory(i), algorithm, budgets[i]);
    });
    return kept;
}

struct Arm {
    std::string name;
    double share;
    KeptSets kept;
};

// PARTNER: RDP with weight 1 + lambda for trajectories that have a join partner
std::vector<Arm> baseline_arms(const core::TrajectoryDb &db,
                               const std::vector<teacher::JoinPair> &pairs, bool all_join_pairs,
                               std::span<const double> shares) {
    using selection::AlgorithmType;
    std::vector<char> partnered(db.size(), 0);
    for (const auto &pair : pairs)
        if (all_join_pairs || pair.seen)
            partnered[pair.i] = partnered[pair.j] = 1;
    std::vector<Arm> arms;
    for (const double share : shares) {
        const auto budget = allocation::total_budget(db, share);
        auto geometric = [&](double alpha) {
            return allocation::create_trajectory_budgets(
                db, allocation::AllocatorType::SizeProportional, budget, alpha);
        };
        const auto equal = geometric(0.0);
        arms.push_back({"RDP_a0", share, select(db, AlgorithmType::RDP, equal)});
        arms.push_back({"RDP_a025", share, select(db, AlgorithmType::RDP, geometric(kRdpAlpha))});
        arms.push_back({"SED_a0", share, select(db, AlgorithmType::SED, equal)});
        arms.push_back({"UNI_a0", share, select(db, AlgorithmType::UNIFORM, equal)});
        for (const double lambda : kPartnerLambdas) {
            std::vector<double> weights(db.size());
            for (std::size_t i = 0; i < db.size(); ++i)
                weights[i] = 1.0 + (partnered[i] ? lambda : 0.0);
            arms.push_back({std::format("PARTNER_l{:g}", lambda), share,
                            select(db, AlgorithmType::RDP,
                                   weighted_budgets(db, weights, budget - 2 * db.size()))});
        }
    }
    return arms;
}

void validate(const Arm &arm, const core::TrajectoryDb &db) {
    if (arm.kept.size() != db.size())
        throw std::runtime_error(arm.name + ": wrong number of trajectories");
    std::size_t total = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto &kept = arm.kept[i];
        const auto n = db.trajectory(i).size();
        if (kept.size() < 2 || kept.front() != 0 || kept.back() != n - 1 ||
            !std::ranges::is_sorted(kept) || std::ranges::adjacent_find(kept) != kept.end())
            throw std::runtime_error(
                std::format("{}: invalid kept set for trajectory {}", arm.name, i));
        total += kept.size();
    }
    const auto budget = allocation::total_budget(db, arm.share);
    if (total != budget)
        throw std::runtime_error(std::format("{}: {} points, budget {}", arm.name, total, budget));
}

struct PairRecall {
    double seen = kNan, unseen = kNan;
};

// Mean per-trajectory recall on the seen and unseen halves of the teacher's join pairs
PairRecall pair_recall(const core::TrajectoryDb &compressed,
                       const std::vector<teacher::JoinPair> &pairs, const query::JoinConfig &join) {
    std::vector<char> matched(pairs.size());
    core::parallel_for(pairs.size(), [&](std::size_t k) {
        matched[k] =
            query::frechet_within(compressed.trajectory(pairs[k].i),
                                  compressed.trajectory(pairs[k].j), join.eps_m, join.frechet);
    });
    PairRecall result;
    for (const bool seen : {true, false}) {
        std::vector<std::size_t> total(compressed.size(), 0), hits(compressed.size(), 0);
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            if (pairs[k].seen != seen)
                continue;
            for (const auto t : {pairs[k].i, pairs[k].j}) {
                ++total[t];
                hits[t] += matched[k];
            }
        }
        double sum = 0;
        std::size_t count = 0;
        for (std::size_t t = 0; t < total.size(); ++t)
            if (total[t] != 0) {
                sum += static_cast<double>(hits[t]) / static_cast<double>(total[t]);
                ++count;
            }
        (seen ? result.seen : result.unseen) = count ? sum / static_cast<double>(count) : kNan;
    }
    return result;
}

// Whether the teacher trained on the join at eps_m in mode: one of its thresholds in its own join
// mode, or its held-out term at join.eps_m
bool trained_on(const pipeline::Config &config, const teacher::ObjectiveConfig &objective,
                double eps_m, query::JoinMode mode) {
    const auto &extra = config.teacher.extra_join_eps_m;
    const bool teacher_eps =
        eps_m == config.join.eps_m || std::ranges::find(extra, eps_m) != extra.end();
    const bool held_out_term =
        objective.held_out_term && mode == query::JoinMode::HeldOut && eps_m == config.join.eps_m;
    return (teacher_eps && mode == objective.join_mode) || held_out_term;
}

std::ofstream open_csv(const std::filesystem::path &file, const std::string &header) {
    const bool fresh = !std::filesystem::exists(file);
    std::ofstream out(file, std::ios::app);
    if (fresh)
        out << header << '\n';
    return out;
}

} // namespace

int main(int argc, char **argv) try {
    const auto args = parse_args(argc, argv);
    const auto base_config = effective_config(args);
    std::filesystem::create_directories(args.out);
    const auto all = io::load_trajectories(base_config.data_path);
    std::cerr << std::format("loaded {} trajectories, {} points\n", all.size(), all.num_points());

    auto results = open_csv(
        args.out / "results.csv",
        "seed,share,arm,budget,knn_recall,range_f1,where_at_avg_hit,join_f1,clustering,"
        "where_at_within_100,join_micro_f1,knn_tau,norm_knn,norm_range,norm_where_at,norm_join,"
        "norm_clustering,J4,J_T,join_recall_seen,join_recall_unseen,predicted_clusters,"
        "eval_s,join_mode,self_join_f1,norm_self_join,teacher_join_mode,norm_join_main,"
        "norm_clustering_main,self_knn_recall,norm_self_knn,norm_self_join_mean,J4_self,"
        "teacher_knn_query");
    auto floors =
        open_csv(args.out / "floors.csv",
                 "seed,knn_recall,range_f1,where_at_avg_hit,join_f1,clustering,self_join_f1,"
                 "self_knn_recall");
    auto thresholds =
        open_csv(args.out / "thresholds.csv",
                 "seed,share,arm,join,eps_m,join_f1,norm_join,recall,teacher_threshold");
    auto terms = open_csv(args.out / "terms.csv",
                          "seed,share,arm,teacher_knn,teacher_range,teacher_where_at,teacher_join,"
                          "norm_knn,norm_range,norm_where_at,norm_join,norm_self_join");
    auto budgets = open_csv(args.out / "budgets.csv", "seed,share,arm,points,budget");
    auto timings = open_csv(args.out / "timings.csv", "seed,step,seconds");
    std::ofstream knn_check;
    if (args.knn_check)
        knn_check = open_csv(args.out / "knn_check.csv",
                             "seed,share,arm,proxy_recall,db_recall,test_recall,norm_proxy,"
                             "norm_db,norm_test");

    for (const auto seed : args.seeds.value_or(std::vector{base_config.seed})) {
        auto config = base_config;
        config.seed = seed;
        const auto parts = split::split(all, config.split, seed);
        const auto db = core::subset(all, parts.database);
        std::cerr << std::format("seed {}: database {} trajectories, {} points\n", seed, db.size(),
                                 db.num_points());

        // Database only
        auto start = std::chrono::steady_clock::now();
        auto workloads = teacher::load_or_build_workloads(
            db, pipeline::teacher_workload_config(config), seed, config.teacher.cache_dir);

        if (args.export_db) {
            const auto dir = *args.export_db / std::format("seed{}", seed);
            io::export_database(dir, db);
            for (const double share : config.budget_shares)
                io::export_budgets(dir, std::format("{:g}pct", share * 100),
                                   allocation::create_trajectory_budgets(
                                       db, allocation::AllocatorType::SizeProportional,
                                       allocation::total_budget(db, share), 0.0));
            const auto &practice = workloads.range.queries;
            std::vector<std::size_t> order(practice.size());
            std::iota(order.begin(), order.end(), std::size_t{0});
            std::mt19937_64 rng(seed + kExportBoxSeedOffset);
            std::ranges::shuffle(order, rng);
            order.resize(std::min(order.size(), kExportBoxes));
            std::vector<query::RangeQuery> boxes;
            for (const auto k : order)
                boxes.push_back(practice[k]);
            const auto hits = io::export_range_hits(dir, db, boxes);
            std::cerr << std::format("seed {}: database written to {}, {} of {} practice range "
                                     "boxes cover {} of {} points\n",
                                     seed, dir.string(), boxes.size(), practice.size(), hits,
                                     db.num_points());
            continue;
        }
        const auto pairs = workloads.positives;
        const teacher::Teacher model(db, std::move(workloads),
                                     pipeline::teacher_objective_config(config), seed);
        timings << std::format("{},teacher_setup,{:.2f}\n", seed, core::seconds_since(start));

        std::vector<Arm> arms;
        if (!args.arms.empty()) {
            for (const double share : config.budget_shares) {
                start = std::chrono::steady_clock::now();
                for (auto &arm : model.run(allocation::total_budget(db, share), args.arms))
                    arms.push_back({arm.name, share, core::to_kept(arm.simplification)});
                timings << std::format("{},teacher_{:g},{:.2f}\n", seed, share,
                                       core::seconds_since(start));
            }
        }
        if (args.baselines) {
            for (auto &arm :
                 baseline_arms(db, pairs, config.teacher.all_join_pairs, config.budget_shares))
                arms.push_back(std::move(arm));
            for (const double share : config.budget_shares)
                for (auto &arm : model.single_recipes(allocation::total_budget(db, share)))
                    arms.push_back({arm.name, share, core::to_kept(arm.simplification)});
        }
        for (const auto &name : args.external)
            for (const double share : config.budget_shares) {
                const auto file = *args.external_dir / std::format("seed{}", seed) /
                                  std::format("{}_{:g}pct.bin", name, share * 100);
                arms.push_back({name, share, io::read_selection(file, db)});
            }
        for (const auto &arm : arms)
            validate(arm, db);

        // Held-out data from here on
        start = std::chrono::steady_clock::now();
        const auto pool = core::subset(all, parts.queries);
        const pipeline::TestWorkloads test_workloads(
            db, pool, config,
            teacher::self_knn_workload(model.objective().workloads(), config.knn.p));
        // Both joins at other thresholds: does the join gain hold beyond the ones trained on?
        std::vector<query::JoinWorkload> other_joins;
        for (const double eps : args.eval_eps) {
            other_joins.push_back(query::generate_self_join(db, eps, config.join.frechet));
            other_joins.push_back(
                query::generate_join_queries(db, pool,
                                             {.query_count = config.join.query_count,
                                              .eps_m = eps,
                                              .mode = query::JoinMode::HeldOut,
                                              .frechet = config.join.frechet},
                                             seed));
        }
        timings << std::format("{},test_workloads,{:.2f}\n", seed, core::seconds_since(start));

        const auto test = test_workloads.view();
        const auto &scoring = config.where_at_scoring;
        const auto floor_db = core::materialize(db, core::endpoints_only(db));
        const auto floor = evaluator::Evaluator(db, floor_db, test, scoring).evaluate().scores;
        floors << std::format("{},{},{},{},{},{},{},{}\n", seed, floor.knn.recall_at_k,
                              floor.range_f1, floor.where_at.average_hit, floor.join.f1,
                              floor.clustering.score, floor.self_join.f1,
                              floor.self_knn.recall_at_k);
        std::vector<double> other_floor;
        for (const auto &workload : other_joins)
            other_floor.push_back(evaluator::evaluate_join(floor_db, workload).f1);
        // Route clustering at each threshold, built on the self join there
        std::vector<query::RouteClusteringWorkload> other_clusters;
        std::vector<double> cluster_floor;
        if (config.clustering.enabled)
            for (const auto &workload : other_joins)
                if (workload.mode == query::JoinMode::Self) {
                    auto clustering = config.clustering;
                    clustering.eps_m = workload.eps_m;
                    other_clusters.push_back(
                        query::generate_route_clustering(db, clustering, &workload));
                    cluster_floor.push_back(
                        evaluator::evaluate_route_clustering(floor_db, other_clusters.back())
                            .score);
                }

        const auto &objective = model.objective();
        const double db_knn_floor = args.knn_check ? objective.knn_recall(floor_db) : kNan;
        for (const auto &arm : arms) {
            start = std::chrono::steady_clock::now();
            const auto compressed = core::materialize(db, arm.kept);
            const auto scores =
                evaluator::Evaluator(db, compressed, test, scoring).evaluate().scores;
            const auto normalized = evaluator::normalize_scores(scores, floor);
            const auto recall = pair_recall(compressed, pairs, config.join);
            results << std::format(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2f},"
                "{},{},{},{},{},{},{},{},{},{},{}\n",
                seed, arm.share, arm.name, allocation::total_budget(db, arm.share),
                scores.knn.recall_at_k, scores.range_f1, scores.where_at.average_hit,
                scores.join.f1, scores.clustering.score, scores.where_at.within_radius,
                scores.join.micro_f1, scores.knn.kendalls_tau_topk, normalized.knn,
                normalized.range, normalized.where_at, normalized.join, normalized.clustering,
                normalized.combined, objective.total(arm.kept), recall.seen, recall.unseen,
                scores.clustering.predicted_clusters, core::seconds_since(start),
                mode_name(config.join.mode), scores.self_join.f1, normalized.self_join,
                mode_name(objective.config().join_mode), normalized.join_f1,
                normalized.clustering_score, scores.self_knn.recall_at_k, normalized.self_knn,
                normalized.self_join_mean, normalized.combined_self,
                objective.config().compressed_knn_query ? "Compressed" : "Original");

            for (std::size_t k = 0; k < other_joins.size(); ++k) {
                const auto &workload = other_joins[k];
                const auto join_scores = evaluator::evaluate_join(compressed, workload);
                const bool teacher_threshold =
                    trained_on(config, objective.config(), workload.eps_m, workload.mode);
                thresholds << std::format("{},{},{},{},{},{},{},{},{}\n", seed, arm.share, arm.name,
                                          mode_name(workload.mode), workload.eps_m, join_scores.f1,
                                          (join_scores.f1 - other_floor[k]) /
                                              (1.0 - other_floor[k]),
                                          join_scores.recall, teacher_threshold ? 1 : 0);
            }
            for (std::size_t k = 0; k < other_clusters.size(); ++k) {
                const auto &workload = other_clusters[k];
                const double score =
                    evaluator::evaluate_route_clustering(compressed, workload).score;
                // Clustering is built on the self join at its threshold
                const bool teacher_threshold =
                    trained_on(config, objective.config(), workload.eps_m, query::JoinMode::Self);
                thresholds << std::format("{},{},{},Clustering,{},{},{},,{}\n", seed, arm.share,
                                          arm.name, workload.eps_m, score,
                                          (score - cluster_floor[k]) / (1.0 - cluster_floor[k]),
                                          teacher_threshold ? 1 : 0);
            }
            // The teacher's own estimate of each metric, next to the test's
            const auto teacher_metrics = objective.metrics(arm.kept);
            const auto &teacher_floors = objective.floors();
            auto norm = [&](teacher::Objective::Term t) {
                return (teacher_metrics[t] - teacher_floors[t]) / (1.0 - teacher_floors[t]);
            };
            terms << std::format("{},{},{},{},{},{},{},{},{},{},{},{}\n", seed, arm.share, arm.name,
                                 norm(teacher::Objective::Knn), norm(teacher::Objective::Range),
                                 norm(teacher::Objective::WhereAt), norm(teacher::Objective::Join),
                                 normalized.knn, normalized.range, normalized.where_at,
                                 normalized.join, normalized.self_join);
            if (args.knn_check) {
                const double db_knn = objective.knn_recall(compressed);
                knn_check << std::format("{},{},{},{},{},{},{},{},{}\n", seed, arm.share, arm.name,
                                         teacher_metrics[teacher::Objective::Knn], db_knn,
                                         scores.knn.recall_at_k, norm(teacher::Objective::Knn),
                                         (db_knn - db_knn_floor) / (1.0 - db_knn_floor),
                                         normalized.knn);
                knn_check.flush();
            }
            if (teacher::is_arm(arm.name))
                for (std::size_t i = 0; i < arm.kept.size(); ++i)
                    budgets << std::format("{},{},{},{},{}\n", seed, arm.share, arm.name,
                                           db.trajectory(i).size(), arm.kept[i].size());
            for (auto *out : {&results, &thresholds, &terms, &budgets})
                out->flush();
            std::cerr << std::format(
                "seed {} {} {:g}: J4 {:.4f}, J4-self {:.4f}, join {:.4f}, self join {:.4f}\n", seed,
                arm.name, arm.share, normalized.combined, normalized.combined_self, normalized.join,
                normalized.self_join);
        }
        timings.flush();
        floors.flush();
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}

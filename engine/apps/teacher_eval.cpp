// Runs the teacher and the baselines on several seeds and scores every arm on the held-out test
// workloads. The teacher sees the database only; test workloads are built after every arm is done.
// Split, workload, scoring and teacher settings come from the same JSON config as `run`
// (--config), so both measure the same thing; the other flags override parts of it.
// J4's join term follows join.mode, and so does the teacher's own join term unless teacher.join or
// --teacher-join says otherwise (teacher_join_mode column). The database self join is scored either
// way (self_join_f1, norm_self_join, the seen/unseen recall and thresholds.csv): its pairs are the
// teacher's training pairs, so with join.mode HeldOut it is the in-sample comparison and stays out
// of J4.
// --train-eps adds self-join thresholds to the teacher's join term, --held-out-term the held-out
// form next to a self join, and --eval-eps sets the thresholds in thresholds.csv, where the self
// join, the held-out join and route clustering (when enabled) are scored at each.
// Writes results.csv, floors.csv, thresholds.csv, terms.csv, budgets.csv and timings.csv to --out
// (and knn_check.csv with --knn-check); summarise with teacher_stats.py.
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/evaluator/combined.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/join.hpp"
#include "backend/engine/io/parquet_loader.hpp"
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
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using teacher::KeptSets;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
// Unset options keep the config's value. Arms default to every teacher arm, not the config's.
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
    std::optional<std::vector<double>> term_weights; // config: teacher.term_weights
    std::vector<double> eval_eps{100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0};
    bool knn_check = false;
};

template <typename T> std::vector<T> parse_list(const std::string &s) {
    std::vector<T> r;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) {
        if constexpr (std::is_same_v<T, std::string>)
            r.push_back(item);
        else
            r.push_back(static_cast<T>(std::stod(item)));
    }
    return r;
}

Args parse_args(int argc, char **argv) {
    Args a;
    for (int k = 1; k < argc; ++k) {
        const std::string key = argv[k];
        auto value = [&] {
            if (k + 1 >= argc)
                throw std::invalid_argument("missing value for " + key);
            return std::string(argv[++k]);
        };
        if (key == "--config")
            a.config = value();
        else if (key == "--data")
            a.data = value();
        else if (key == "--out")
            a.out = value();
        else if (key == "--cache")
            a.cache = value();
        else if (key == "--seeds")
            a.seeds = parse_list<std::uint64_t>(value());
        else if (key == "--shares")
            a.shares = parse_list<double>(value());
        else if (key == "--arms")
            a.arms = parse_list<std::string>(value());
        else if (key == "--train-eps")
            a.train_eps = parse_list<double>(value());
        else if (key == "--eval-eps")
            a.eval_eps = parse_list<double>(value());
        else if (key == "--term-weights") {
            a.term_weights = parse_list<double>(value());
            if (a.term_weights->size() != 4)
                throw std::invalid_argument("--term-weights takes kNN,range,where-at,join");
        } else if (key == "--independent-knn")
            a.ranking_knn = false;
        else if (key == "--held-out-term")
            a.held_out_term = true;
        else if (key == "--knn-check")
            a.knn_check = true;
        else if (key == "--teacher-join") {
            const auto mode = value();
            if (mode != "HeldOut" && mode != "Self")
                throw std::invalid_argument("--teacher-join must be HeldOut or Self");
            a.teacher_join = mode == "Self" ? query::JoinMode::Self : query::JoinMode::HeldOut;
        } else if (key == "--no-baselines")
            a.baselines = false;
        else if (key == "--seen-pairs-only")
            a.all_join_pairs = false;
        else
            throw std::invalid_argument("unknown argument " + key +
                                        " (--config --data --out --cache --seeds --shares --arms "
                                        "--teacher-join --train-eps --held-out-term "
                                        "--independent-knn --term-weights --eval-eps --knn-check "
                                        "--no-baselines --seen-pairs-only)");
    }
    for (const auto &arm : a.arms)
        if (!teacher::is_arm(arm))
            throw std::invalid_argument("unknown teacher arm " + arm);
    return a;
}

// The config with the command line overrides applied
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
    if (args.term_weights)
        std::ranges::copy(*args.term_weights, config.teacher.term_weights.begin());
    // The self join is always scored: with a held-out join in J4 it is the in-sample comparison
    config.self_join = true;
    return config;
}

const char *mode_name(query::JoinMode mode) {
    return mode == query::JoinMode::Self ? "Self" : "HeldOut";
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::size_t total_budget(const core::TrajectoryDb &db, double share) {
    const std::size_t endpoints = 2 * db.size();
    const double extra = share * static_cast<double>(db.num_points() - endpoints);
    return endpoints + static_cast<std::size_t>(std::llround(extra));
}

core::Simplification to_simplification(const KeptSets &kept) {
    core::Simplification s;
    for (const auto &k : kept)
        s.add_trajectory(k);
    return s;
}

KeptSets to_kept(const core::Simplification &s) {
    KeptSets kept(s.size());
    for (std::size_t i = 0; i < s.size(); ++i)
        kept[i].assign(s.trajectory(i).begin(), s.trajectory(i).end());
    return kept;
}

KeptSets endpoints_only(const core::TrajectoryDb &db) {
    KeptSets kept(db.size());
    for (std::size_t i = 0; i < db.size(); ++i)
        kept[i] = {0, db.trajectory(i).size() - 1};
    return kept;
}

// Interior points shared in proportion to w, capped at each trajectory's interior, the rest by
// largest remainder. Budgets include the endpoints
std::vector<std::size_t> weighted_budgets(const core::TrajectoryDb &db,
                                          const std::vector<double> &w,
                                          std::size_t interior_total) {
    const std::size_t n = db.size();
    std::vector<std::size_t> cap(n), q(n, 0);
    for (std::size_t i = 0; i < n; ++i)
        cap[i] = db.trajectory(i).size() >= 2 ? db.trajectory(i).size() - 2 : 0;
    std::vector<char> open(n, 1);
    std::size_t left = interior_total;
    while (left > 0) {
        double sw = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (open[i])
                sw += w[i];
        if (sw <= 0)
            break;
        std::vector<double> frac(n, 0);
        std::size_t given = 0;
        bool capped = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (!open[i])
                continue;
            const double want = static_cast<double>(left) * w[i] / sw;
            auto add = static_cast<std::size_t>(want);
            if (q[i] + add >= cap[i]) {
                add = cap[i] - q[i];
                open[i] = 0;
                capped = true;
            } else {
                frac[i] = want - static_cast<double>(add);
            }
            q[i] += add;
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
            ++q[idx[k]];
            --left;
            if (q[idx[k]] == cap[idx[k]])
                open[idx[k]] = 0;
        }
        if (given == 0 && idx.empty())
            break;
    }
    for (std::size_t i = 0; i < n; ++i)
        q[i] += std::min<std::size_t>(2, db.trajectory(i).size());
    return q;
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

// RDP, SED and UNIFORM with equal points, RDP with alpha 0.25, and PARTNER: weight 1 + lambda for
// trajectories with a join partner (in the pairs the teacher uses), then RDP
std::vector<Arm> baseline_arms(const core::TrajectoryDb &db,
                               const std::vector<teacher::JoinPair> &pairs, bool all_join_pairs,
                               std::span<const double> shares) {
    using selection::AlgorithmType;
    std::vector<char> partnered(db.size(), 0);
    for (const auto &p : pairs)
        if (all_join_pairs || p.seen)
            partnered[p.i] = partnered[p.j] = 1;
    std::vector<Arm> arms;
    for (const double share : shares) {
        const auto budget = total_budget(db, share);
        auto geometric = [&](double alpha) {
            return allocation::create_trajectory_budgets(
                db, allocation::AllocatorType::SizeProportional, budget, alpha);
        };
        const auto equal = geometric(0.0);
        arms.push_back({"RDP_a0", share, select(db, AlgorithmType::RDP, equal)});
        arms.push_back({"RDP_a025", share, select(db, AlgorithmType::RDP, geometric(0.25))});
        arms.push_back({"SED_a0", share, select(db, AlgorithmType::SED, equal)});
        arms.push_back({"UNI_a0", share, select(db, AlgorithmType::UNIFORM, equal)});
        for (const double lambda : {0.5, 1.0}) {
            std::vector<double> w(db.size());
            for (std::size_t i = 0; i < db.size(); ++i)
                w[i] = 1.0 + (partnered[i] ? lambda : 0.0);
            arms.push_back(
                {std::format("PARTNER_l{:g}", lambda), share,
                 select(db, AlgorithmType::RDP, weighted_budgets(db, w, budget - 2 * db.size()))});
        }
    }
    return arms;
}

// Kept sets must be valid and add up to the budget
void validate(const Arm &arm, const core::TrajectoryDb &db) {
    if (arm.kept.size() != db.size())
        throw std::runtime_error(arm.name + ": wrong number of trajectories");
    std::size_t total = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto &k = arm.kept[i];
        const auto n = db.trajectory(i).size();
        if (k.size() < 2 || k.front() != 0 || k.back() != n - 1 || !std::ranges::is_sorted(k) ||
            std::ranges::adjacent_find(k) != k.end())
            throw std::runtime_error(
                std::format("{}: invalid kept set for trajectory {}", arm.name, i));
        total += k.size();
    }
    if (total != total_budget(db, arm.share))
        throw std::runtime_error(
            std::format("{}: {} points, budget {}", arm.name, total, total_budget(db, arm.share)));
}

struct PairRecall {
    double seen = kNan, unseen = kNan;
};

// Recall per trajectory on the seen and unseen halves of the teacher's join pairs, averaged over
// trajectories
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

// Recall@k of the teacher's own kNN queries (database trajectories, themselves excluded) answered
// on the compressed database, so competitors are compressed too, unlike the teacher's kNN term
double db_knn_recall(const core::TrajectoryDb &db, const core::TrajectoryDb &compressed,
                     const teacher::TeacherWorkloads &w) {
    std::vector<double> recall(db.size(), kNan);
    core::parallel_for(db.size(), [&](std::size_t e) {
        const auto &q = w.knn[e];
        if (!q.valid)
            return;
        auto found = query::knn(compressed, db.trajectory(e), w.k + 1, w.frechet);
        std::erase(found, e);
        found.resize(std::min(found.size(), w.k));
        std::size_t hits = 0;
        for (const auto i : q.top)
            hits += std::ranges::find(found, std::size_t{i}) != found.end();
        recall[e] = static_cast<double>(hits) / static_cast<double>(w.k);
    });
    double sum = 0;
    std::size_t count = 0;
    for (const double r : recall)
        if (!std::isnan(r)) {
            sum += r;
            ++count;
        }
    return count ? sum / static_cast<double>(count) : kNan;
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
        "norm_clustering_main");
    auto floors =
        open_csv(args.out / "floors.csv",
                 "seed,knn_recall,range_f1,where_at_avg_hit,join_f1,clustering,self_join_f1");
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
        const auto pairs = workloads.positives;
        const teacher::Teacher model(db, std::move(workloads),
                                     pipeline::teacher_objective_config(config), seed);
        timings << std::format("{},teacher_setup,{:.2f}\n", seed, seconds_since(start));

        std::vector<Arm> arms;
        for (const double share : config.budget_shares) {
            start = std::chrono::steady_clock::now();
            for (auto &arm : model.run(total_budget(db, share), args.arms))
                arms.push_back({arm.name, share, to_kept(arm.simplification)});
            timings << std::format("{},teacher_{:g},{:.2f}\n", seed, share, seconds_since(start));
        }
        if (args.baselines) {
            for (auto &arm :
                 baseline_arms(db, pairs, config.teacher.all_join_pairs, config.budget_shares))
                arms.push_back(std::move(arm));
            for (const double share : config.budget_shares)
                for (auto &arm : model.single_recipes(total_budget(db, share)))
                    arms.push_back({arm.name, share, to_kept(arm.simplification)});
        }
        for (const auto &arm : arms)
            validate(arm, db);

        // Held-out data from here on
        start = std::chrono::steady_clock::now();
        const auto pool = core::subset(all, parts.queries);
        const pipeline::TestWorkloads test_workloads(db, pool, config);
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
        timings << std::format("{},test_workloads,{:.2f}\n", seed, seconds_since(start));

        const auto test = test_workloads.view();
        const auto &scoring = config.where_at_scoring;
        const auto floor_db = core::materialize(db, to_simplification(endpoints_only(db)));
        const auto floor = evaluator::Evaluator(db, floor_db, test, scoring).evaluate().scores;
        floors << std::format("{},{},{},{},{},{},{}\n", seed, floor.knn.recall_at_k, floor.range_f1,
                              floor.where_at.average_hit, floor.join.f1, floor.clustering.score,
                              floor.self_join.f1);
        std::vector<double> other_floor;
        for (const auto &w : other_joins)
            other_floor.push_back(evaluator::evaluate_join(floor_db, w).f1);
        // Route clustering at each threshold, built on the self join there
        std::vector<query::RouteClusteringWorkload> other_clusters;
        std::vector<double> cluster_floor;
        if (config.clustering.enabled)
            for (const auto &w : other_joins)
                if (w.mode == query::JoinMode::Self) {
                    auto clustering = config.clustering;
                    clustering.eps_m = w.eps_m;
                    other_clusters.push_back(query::generate_route_clustering(db, clustering, &w));
                    cluster_floor.push_back(
                        evaluator::evaluate_route_clustering(floor_db, other_clusters.back())
                            .score);
                }

        const auto &objective = model.objective();
        const double db_knn_floor =
            args.knn_check ? db_knn_recall(db, floor_db, objective.workloads()) : kNan;
        for (const auto &arm : arms) {
            start = std::chrono::steady_clock::now();
            const auto compressed = core::materialize(db, to_simplification(arm.kept));
            const auto s = evaluator::Evaluator(db, compressed, test, scoring).evaluate().scores;
            const auto n = evaluator::normalize_scores(s, floor);
            const auto recall = pair_recall(compressed, pairs, config.join);
            results << std::format(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2f},"
                "{},{},{},{},{},{}\n",
                seed, arm.share, arm.name, total_budget(db, arm.share), s.knn.recall_at_k,
                s.range_f1, s.where_at.average_hit, s.join.f1, s.clustering.score,
                s.where_at.within_radius, s.join.micro_f1, s.knn.kendalls_tau_topk, n.knn, n.range,
                n.where_at, n.join, n.clustering, n.combined, objective.total(arm.kept),
                recall.seen, recall.unseen, s.clustering.predicted_clusters, seconds_since(start),
                mode_name(config.join.mode), s.self_join.f1, n.self_join,
                mode_name(objective.config().join_mode), n.join_f1, n.clustering_score);

            for (std::size_t k = 0; k < other_joins.size(); ++k) {
                const auto &w = other_joins[k];
                const auto js = evaluator::evaluate_join(compressed, w);
                // Trained on: one of the teacher's thresholds in the teacher's join mode, or the
                // held-out term at join.eps_m
                const auto &extra = config.teacher.extra_join_eps_m;
                const bool teacher_eps = w.eps_m == config.join.eps_m ||
                                         std::ranges::find(extra, w.eps_m) != extra.end();
                const bool teacher_threshold =
                    (teacher_eps && w.mode == objective.config().join_mode) ||
                    (w.mode == query::JoinMode::HeldOut && objective.config().held_out_term &&
                     w.eps_m == config.join.eps_m);
                thresholds << std::format("{},{},{},{},{},{},{},{},{}\n", seed, arm.share, arm.name,
                                          mode_name(w.mode), w.eps_m, js.f1,
                                          (js.f1 - other_floor[k]) / (1.0 - other_floor[k]),
                                          js.recall, teacher_threshold ? 1 : 0);
            }
            for (std::size_t k = 0; k < other_clusters.size(); ++k) {
                const auto &w = other_clusters[k];
                const double score = evaluator::evaluate_route_clustering(compressed, w).score;
                const auto &extra = config.teacher.extra_join_eps_m;
                const bool teacher_threshold =
                    objective.config().join_mode == query::JoinMode::Self &&
                    (w.eps_m == config.join.eps_m ||
                     std::ranges::find(extra, w.eps_m) != extra.end());
                thresholds << std::format("{},{},{},Clustering,{},{},{},,{}\n", seed, arm.share,
                                          arm.name, w.eps_m, score,
                                          (score - cluster_floor[k]) / (1.0 - cluster_floor[k]),
                                          teacher_threshold ? 1 : 0);
            }
            // The teacher's own estimate of each metric, next to the test's. Its join term is
            // like-for-like with norm_join or norm_self_join, whichever its join mode is
            const auto m = objective.metrics(arm.kept);
            const auto &f = objective.floors();
            auto norm = [&](teacher::Objective::Term t) { return (m[t] - f[t]) / (1.0 - f[t]); };
            terms << std::format("{},{},{},{},{},{},{},{},{},{},{},{}\n", seed, arm.share, arm.name,
                                 norm(teacher::Objective::Knn), norm(teacher::Objective::Range),
                                 norm(teacher::Objective::WhereAt), norm(teacher::Objective::Join),
                                 n.knn, n.range, n.where_at, n.join, n.self_join);
            if (args.knn_check) {
                const double db_knn = db_knn_recall(db, compressed, objective.workloads());
                knn_check << std::format("{},{},{},{},{},{},{},{},{}\n", seed, arm.share, arm.name,
                                         m[teacher::Objective::Knn], db_knn, s.knn.recall_at_k,
                                         norm(teacher::Objective::Knn),
                                         (db_knn - db_knn_floor) / (1.0 - db_knn_floor), n.knn);
                knn_check.flush();
            }
            if (teacher::is_arm(arm.name))
                for (std::size_t i = 0; i < arm.kept.size(); ++i)
                    budgets << std::format("{},{},{},{},{}\n", seed, arm.share, arm.name,
                                           db.trajectory(i).size(), arm.kept[i].size());
            for (auto *out : {&results, &thresholds, &terms, &budgets})
                out->flush();
            std::cerr << std::format("seed {} {} {:g}: J4 {:.4f}, join {:.4f}, self join {:.4f}\n",
                                     seed, arm.name, arm.share, n.combined, n.join, n.self_join);
        }
        timings.flush();
        floors.flush();
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}

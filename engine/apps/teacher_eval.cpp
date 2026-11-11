// Runs the teacher and the baselines on several seeds and scores every arm on the held-out test
// workloads. The teacher sees the database only; test workloads are built after every arm is done.
// Writes results.csv, floors.csv, thresholds.csv, terms.csv, budgets.csv and timings.csv to --out;
// summarise with teacher_stats.py.
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/evaluator/combined.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/join.hpp"
#include "backend/engine/io/parquet_loader.hpp"
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using teacher::KeptSets;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kJoinEps = 150.0;
constexpr auto kFrechet = query::FrechetVariant::Continuous;

struct Args {
    std::filesystem::path data = "data/geolife.parquet";
    std::filesystem::path out = "results/teacher";
    std::filesystem::path cache = "cache";
    std::vector<std::uint64_t> seeds{42};
    std::vector<double> shares{0.0025, 0.005, 0.01};
    std::vector<std::string> arms{teacher::kArms.begin(), teacher::kArms.end()};
    bool baselines = true;
    bool all_join_pairs = true;
    double shape_weight = 0.0;
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
        if (key == "--data")
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
        else if (key == "--shape-weight")
            a.shape_weight = std::stod(value());
        else if (key == "--no-baselines")
            a.baselines = false;
        else if (key == "--seen-pairs-only")
            a.all_join_pairs = false;
        else
            throw std::invalid_argument(
                "unknown argument " + key +
                " (--data --out --cache --seeds --shares --arms --shape-weight --no-baselines "
                "--seen-pairs-only)");
    }
    for (const auto &arm : a.arms)
        if (!teacher::is_arm(arm))
            throw std::invalid_argument("unknown teacher arm " + arm);
    return a;
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

// Recall per trajectory on the seen and unseen halves of the 150 m pairs, averaged over
// trajectories
PairRecall pair_recall(const core::TrajectoryDb &compressed,
                       const std::vector<teacher::JoinPair> &pairs) {
    std::vector<char> matched(pairs.size());
    core::parallel_for(pairs.size(), [&](std::size_t k) {
        matched[k] = query::frechet_within(compressed.trajectory(pairs[k].i),
                                           compressed.trajectory(pairs[k].j), kJoinEps, kFrechet);
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
    std::filesystem::create_directories(args.out);
    const auto all = io::load_trajectories(args.data);
    std::cerr << std::format("loaded {} trajectories, {} points\n", all.size(), all.num_points());

    auto results = open_csv(
        args.out / "results.csv",
        "seed,share,arm,budget,knn_recall,range_f1,where_at_avg_hit,join_f1,clustering,"
        "where_at_within_100,join_micro_f1,knn_tau,norm_knn,norm_range,norm_where_at,norm_join,"
        "norm_clustering,J5,J_fam,J_T,join_recall_seen,join_recall_unseen,predicted_clusters,"
        "eval_s");
    auto floors = open_csv(args.out / "floors.csv",
                           "seed,knn_recall,range_f1,where_at_avg_hit,join_f1,clustering");
    auto thresholds =
        open_csv(args.out / "thresholds.csv", "seed,share,arm,eps_m,join_f1,norm_join,recall");
    auto terms = open_csv(args.out / "terms.csv",
                          "seed,share,arm,teacher_knn,teacher_range,teacher_where_at,teacher_join,"
                          "norm_knn,norm_range,norm_where_at,norm_join");
    auto budgets = open_csv(args.out / "budgets.csv", "seed,share,arm,points,budget");
    auto timings = open_csv(args.out / "timings.csv", "seed,step,seconds");

    for (const auto seed : args.seeds) {
        const auto parts = split::split(all, {split::SplitType::Random, 0.15}, seed);
        const auto db = core::subset(all, parts.database);
        std::cerr << std::format("seed {}: database {} trajectories, {} points\n", seed, db.size(),
                                 db.num_points());

        // Database only
        auto start = std::chrono::steady_clock::now();
        auto workloads =
            teacher::load_or_build_workloads(db,
                                             {.k = 10,
                                              .join_eps_m = kJoinEps,
                                              .frechet = kFrechet,
                                              .range = {64000, 2000.0, 604800, 5, 20, 1000},
                                              .where_at = {128, 60.0}},
                                             seed, args.cache);
        const auto pairs = workloads.positives;
        const teacher::Teacher model(
            db, std::move(workloads),
            {.all_join_pairs = args.all_join_pairs, .shape_weight = args.shape_weight}, seed);
        timings << std::format("{},teacher_setup,{:.2f}\n", seed, seconds_since(start));

        std::vector<Arm> arms;
        for (const double share : args.shares) {
            start = std::chrono::steady_clock::now();
            for (auto &arm : model.run(total_budget(db, share), args.arms))
                arms.push_back({arm.name, share, to_kept(arm.simplification)});
            timings << std::format("{},teacher_{:g},{:.2f}\n", seed, share, seconds_since(start));
        }
        if (args.baselines)
            for (auto &arm : baseline_arms(db, pairs, args.all_join_pairs, args.shares))
                arms.push_back(std::move(arm));
        for (const auto &arm : arms)
            validate(arm, db);

        // Held-out data from here on
        start = std::chrono::steady_clock::now();
        const auto pool = core::subset(all, parts.queries);
        const auto knn = query::generate_knn_queries(
            db, pool, {.k = 10, .query_count = 0, .frechet = kFrechet}, seed);
        const auto where_at = query::generate_where_at_queries(db, {128, 60.0}, seed);
        const auto range =
            query::generate_range_queries(db, {20000, 2000.0, 604800, 5, 20, 1000}, seed);
        const auto join = query::generate_self_join(db, kJoinEps, kFrechet);
        const auto clustering =
            query::generate_route_clustering(db, {true, kJoinEps, 5, kFrechet}, &join);
        // The join at thresholds the teacher does not optimise: does its join gain generalise?
        std::vector<query::JoinWorkload> other_joins;
        for (const double eps : {100.0, 300.0})
            other_joins.push_back(query::generate_self_join(db, eps, kFrechet));
        timings << std::format("{},test_workloads,{:.2f}\n", seed, seconds_since(start));

        const evaluator::Workloads test{.knn = &knn,
                                        .where_at = where_at.queries,
                                        .range = range.queries,
                                        .join = &join,
                                        .clustering = &clustering};
        const evaluator::WhereAtScoringConfig scoring{100.0, 25.0, 500.0};
        const auto floor_db = core::materialize(db, to_simplification(endpoints_only(db)));
        const auto floor = evaluator::Evaluator(db, floor_db, test, scoring).evaluate().scores;
        floors << std::format("{},{},{},{},{},{}\n", seed, floor.knn.recall_at_k, floor.range_f1,
                              floor.where_at.average_hit, floor.join.f1, floor.clustering.score);
        std::vector<double> other_floor;
        for (const auto &w : other_joins)
            other_floor.push_back(evaluator::evaluate_join(floor_db, w).f1);

        const auto &objective = model.objective();
        for (const auto &arm : arms) {
            start = std::chrono::steady_clock::now();
            const auto compressed = core::materialize(db, to_simplification(arm.kept));
            const auto s = evaluator::Evaluator(db, compressed, test, scoring).evaluate().scores;
            const auto n = evaluator::normalize_scores(s, floor);
            const double j_fam =
                (n.knn + n.range + n.where_at + 0.5 * (n.join + n.clustering)) / 4.0;
            const auto recall = pair_recall(compressed, pairs);
            results << std::format(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2f}\n",
                seed, arm.share, arm.name, total_budget(db, arm.share), s.knn.recall_at_k,
                s.range_f1, s.where_at.average_hit, s.join.f1, s.clustering.score,
                s.where_at.within_radius, s.join.micro_f1, s.knn.kendalls_tau_topk, n.knn, n.range,
                n.where_at, n.join, n.clustering, n.combined, j_fam, objective.total(arm.kept),
                recall.seen, recall.unseen, s.clustering.predicted_clusters, seconds_since(start));

            for (std::size_t k = 0; k < other_joins.size(); ++k) {
                const auto js = evaluator::evaluate_join(compressed, other_joins[k]);
                thresholds << std::format(
                    "{},{},{},{},{},{},{}\n", seed, arm.share, arm.name, other_joins[k].eps_m,
                    js.f1, (js.f1 - other_floor[k]) / (1.0 - other_floor[k]), js.recall);
            }
            // The teacher's own estimate of each metric, next to the test's
            const auto m = objective.metrics(arm.kept);
            const auto &f = objective.floors();
            auto norm = [&](teacher::Objective::Term t) { return (m[t] - f[t]) / (1.0 - f[t]); };
            terms << std::format("{},{},{},{},{},{},{},{},{},{},{}\n", seed, arm.share, arm.name,
                                 norm(teacher::Objective::Knn), norm(teacher::Objective::Range),
                                 norm(teacher::Objective::WhereAt), norm(teacher::Objective::Join),
                                 n.knn, n.range, n.where_at, n.join);
            if (teacher::is_arm(arm.name))
                for (std::size_t i = 0; i < arm.kept.size(); ++i)
                    budgets << std::format("{},{},{},{},{}\n", seed, arm.share, arm.name,
                                           db.trajectory(i).size(), arm.kept[i].size());
            for (auto *out : {&results, &thresholds, &terms, &budgets})
                out->flush();
            std::cerr << std::format("seed {} {} {:g}: J5 {:.4f}, J_fam {:.4f}\n", seed, arm.name,
                                     arm.share, n.combined, j_fam);
        }
        timings.flush();
        floors.flush();
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}

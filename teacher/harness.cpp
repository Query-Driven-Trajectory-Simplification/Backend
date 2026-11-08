// Runs the teacher and the baselines, then scores them on the test workloads.
// Held-out data is only touched once every arm is done
#include "teacher.hpp"

#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/evaluator/combined.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/route_clustering.hpp"
#include "backend/engine/split/split.hpp"

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

using teacher::KeptSets;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

struct Args {
    std::filesystem::path data = "data/geolife.parquet";
    std::filesystem::path out = "results";
    std::filesystem::path cache = "cache";
    std::vector<std::uint64_t> seeds{42};
    std::vector<double> shares{0.0025, 0.005, 0.01};
    bool baselines = true;
};

template <typename T> std::vector<T> parse_list(const std::string &s) {
    std::vector<T> r;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');)
        r.push_back(static_cast<T>(std::stod(item)));
    return r;
}

Args parse_args(int argc, char **argv) {
    Args a;
    for (int k = 1; k < argc; ++k) {
        const std::string key = argv[k];
        if (k + 1 >= argc && key != "--no-baselines")
            throw std::invalid_argument("missing value for " + key);
        if (key == "--data")
            a.data = argv[++k];
        else if (key == "--out")
            a.out = argv[++k];
        else if (key == "--cache")
            a.cache = argv[++k];
        else if (key == "--seeds")
            a.seeds = parse_list<std::uint64_t>(argv[++k]);
        else if (key == "--shares")
            a.shares = parse_list<double>(argv[++k]);
        else if (key == "--no-baselines")
            a.baselines = false;
        else
            throw std::invalid_argument("unknown argument " + key);
    }
    return a;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

core::Simplification to_simplification(const KeptSets &kept) {
    core::Simplification s;
    for (const auto &k : kept)
        s.add_trajectory(k);
    return s;
}

KeptSets endpoints_only(const core::TrajectoryDb &db) {
    KeptSets kept(db.size());
    for (std::size_t i = 0; i < db.size(); ++i)
        kept[i] = {0, db.trajectory(i).size() - 1};
    return kept;
}

// Kept sets must be valid and add up to the budget
void validate(const std::string &arm, const core::TrajectoryDb &db, const KeptSets &kept,
              std::size_t budget) {
    if (kept.size() != db.size())
        throw std::runtime_error(arm + ": wrong number of trajectories");
    std::size_t total = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto &k = kept[i];
        const auto n = db.trajectory(i).size();
        if (k.size() < 2 || k.front() != 0 || k.back() != n - 1 || !std::ranges::is_sorted(k) ||
            std::ranges::adjacent_find(k) != k.end())
            throw std::runtime_error(std::format("{}: invalid kept set for trajectory {}", arm, i));
        total += k.size();
    }
    if (total != budget)
        throw std::runtime_error(std::format("{}: {} points, budget {}", arm, total, budget));
}

struct PairRecall {
    double seen = kNan, unseen = kNan;
};

// Pair recall per trajectory, seen and unseen pairs separately
PairRecall pair_recall(const core::TrajectoryDb &compressed, const teacher::TeacherCaches &caches) {
    std::vector<char> matched(caches.pairs.size());
    core::parallel_for(caches.pairs.size(), [&](std::size_t k) {
        const auto &p = caches.pairs[k];
        matched[k] = query::frechet_within(compressed.trajectory(p.i), compressed.trajectory(p.j),
                                           teacher::kJoinEps, query::FrechetVariant::Continuous);
    });
    PairRecall result;
    for (const bool seen : {true, false}) {
        std::vector<std::size_t> pairs(compressed.size(), 0), hits(compressed.size(), 0);
        for (std::size_t k = 0; k < caches.pairs.size(); ++k) {
            const auto &p = caches.pairs[k];
            if (p.seen != seen)
                continue;
            for (const auto t : {p.i, p.j}) {
                ++pairs[t];
                hits[t] += matched[k];
            }
        }
        double sum = 0;
        std::size_t count = 0;
        for (std::size_t t = 0; t < pairs.size(); ++t)
            if (pairs[t] != 0) {
                sum += static_cast<double>(hits[t]) / static_cast<double>(pairs[t]);
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
        "norm_clustering,J5,J_fam,J_T,join_recall_seen,join_recall_unseen,predicted_clusters,eval_"
        "s");
    auto floors = open_csv(args.out / "floors.csv",
                           "seed,knn_recall,range_f1,where_at_avg_hit,join_f1,clustering");
    auto timings = open_csv(args.out / "timings.csv", "seed,step,seconds");

    for (const auto seed : args.seeds) {
        const auto parts = split::split(all, {split::SplitType::Random, 0.15}, seed);
        const auto db = core::subset(all, parts.database);
        std::cerr << std::format("seed {}: database {} trajectories, {} points\n", seed, db.size(),
                                 db.num_points());

        // Database only
        auto start = std::chrono::steady_clock::now();
        const auto caches = teacher::load_or_build_caches(db, seed, args.cache);
        timings << std::format("{},caches,{:.2f}\n", seed, seconds_since(start));
        start = std::chrono::steady_clock::now();
        auto output = teacher::run_teacher(db, caches, seed, args.shares);
        timings << std::format("{},teacher,{:.2f}\n", seed, seconds_since(start));
        for (const auto &[step, s] : output.timings_s)
            timings << std::format("{},teacher:{},{:.2f}\n", seed, step, s);
        auto arms = std::move(output.arms);
        if (args.baselines) {
            start = std::chrono::steady_clock::now();
            for (auto &[name, kept] : teacher::baseline_arms(db, caches, args.shares))
                if (!arms.emplace(name, std::move(kept)).second)
                    throw std::runtime_error("teacher arm collides with baseline " + name);
            timings << std::format("{},baselines,{:.2f}\n", seed, seconds_since(start));
        }

        std::map<std::string, double> share_of;
        for (const double share : args.shares)
            share_of[teacher::share_tag(share)] = share;
        for (const auto &[name, kept] : arms) {
            const auto at = name.rfind("_share");
            if (at == std::string::npos || !share_of.contains(name.substr(at + 1)))
                throw std::runtime_error("arm name without a known _share<x> suffix: " + name);
            validate(name, db, kept, teacher::total_budget(db, share_of.at(name.substr(at + 1))));
        }

        // Held-out data from here on
        start = std::chrono::steady_clock::now();
        const auto pool = core::subset(all, parts.queries);
        const auto knn = query::generate_knn_queries(
            db, pool, {.k = 10, .query_count = 0, .frechet = query::FrechetVariant::Continuous},
            seed);
        const auto where_at = query::generate_where_at_queries(db, {128, 60.0}, seed);
        const auto range =
            query::generate_range_queries(db, {20000, 2000.0, 604800, 5, 20, 1000}, seed);
        const auto join =
            query::generate_self_join(db, teacher::kJoinEps, query::FrechetVariant::Continuous);
        const auto clustering = query::generate_route_clustering(
            db, {true, teacher::kJoinEps, 5, query::FrechetVariant::Continuous}, &join);
        timings << std::format("{},test_workloads,{:.2f}\n", seed, seconds_since(start));
        const evaluator::Workloads workloads{.knn = &knn,
                                             .where_at = where_at.queries,
                                             .range = range.queries,
                                             .join = &join,
                                             .clustering = &clustering};
        const evaluator::WhereAtScoringConfig scoring{100.0, 25.0, 500.0};

        const auto floor_db = core::materialize(db, to_simplification(endpoints_only(db)));
        const auto floor = evaluator::Evaluator(db, floor_db, workloads, scoring).evaluate().scores;
        floors << std::format("{},{},{},{},{},{}\n", seed, floor.knn.recall_at_k, floor.range_f1,
                              floor.where_at.average_hit, floor.join.f1, floor.clustering.score);

        for (const auto &[name, kept] : arms) {
            start = std::chrono::steady_clock::now();
            const auto at = name.rfind("_share");
            const double share = share_of.at(name.substr(at + 1));
            const auto compressed = core::materialize(db, to_simplification(kept));
            const auto s =
                evaluator::Evaluator(db, compressed, workloads, scoring).evaluate().scores;
            const auto n = evaluator::normalize_scores(s, floor);
            const double j_fam =
                (n.knn + n.range + n.where_at + 0.5 * (n.join + n.clustering)) / 4.0;
            const double j_t = output.objective ? output.objective(kept, share) : kNan;
            const auto recall = pair_recall(compressed, caches);
            results << std::format(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2f}\n",
                seed, share, name.substr(0, at), teacher::total_budget(db, share),
                s.knn.recall_at_k, s.range_f1, s.where_at.average_hit, s.join.f1,
                s.clustering.score, s.where_at.within_radius, s.join.micro_f1,
                s.knn.kendalls_tau_topk, n.knn, n.range, n.where_at, n.join, n.clustering,
                n.combined, j_fam, j_t, recall.seen, recall.unseen, s.clustering.predicted_clusters,
                seconds_since(start));
            results.flush();
            std::cerr << std::format("seed {} {}: J5 {:.4f}, J_fam {:.4f}\n", seed, name,
                                     n.combined, j_fam);
        }
        timings.flush();
        floors.flush();
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}

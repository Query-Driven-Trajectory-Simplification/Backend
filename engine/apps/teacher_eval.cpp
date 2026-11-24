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
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/pipeline/test_workloads.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/route_clustering.hpp"
#include "backend/engine/selection/rdp.hpp"
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
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using core::KeptSets;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
// Between the training thresholds, so the teacher never saw them
constexpr std::array kEvalEps{75.0, 125.0, 175.0};
constexpr double kRdpAlpha = 0.25; // RDP_a025
constexpr std::array kPartnerLambdas{0.5, 1.0};
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
    std::optional<std::vector<double>> train_eps;    // config: teacher.extra_join_eps_m
    std::optional<std::vector<double>> term_weights; // config: teacher.term_weights
    std::vector<double> eval_eps{kEvalEps.begin(), kEvalEps.end()};
    bool knn_check = false;
    bool diagnose = false;
    std::size_t knn_outside = 0;               // ranked outsiders per query for the margin term
    std::optional<std::uint64_t> pseudo_split; // re-split the database: tuning without the pool
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
            if (args.term_weights->size() < 4 || args.term_weights->size() > teacher::kTermCount)
                throw std::invalid_argument(
                    "--term-weights takes kNN,range,where-at,join[,shape[,margin]]");
        } else if (key == "--knn-check")
            args.knn_check = true;
        else if (key == "--diagnose")
            args.diagnose = true;
        else if (key == "--knn-outside")
            args.knn_outside = std::stoull(value());
        else if (key == "--pseudo-split")
            args.pseudo_split = std::stoull(value());
        else if (key == "--no-baselines")
            args.baselines = false;
        else
            throw std::invalid_argument("unknown argument " + key +
                                        " (--config --data --out --cache --seeds --shares --arms "
                                        "--train-eps --term-weights --eval-eps --knn-check "
                                        "--diagnose --knn-outside --pseudo-split --no-baselines)");
    }
    teacher::check_arms(args.arms);
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
    if (args.train_eps)
        config.teacher.extra_join_eps_m = *args.train_eps;
    if (args.term_weights) {
        config.teacher.term_weights.fill(0.0);
        std::ranges::copy(*args.term_weights, config.teacher.term_weights.begin());
    }
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
    std::optional<teacher::Objective::Terms> metrics; // the teacher's, if it computed them
};

// PARTNER: RDP with weight 1 + lambda for trajectories that have a join partner
std::vector<Arm> baseline_arms(const core::TrajectoryDb &db,
                               const std::vector<teacher::JoinPair> &pairs,
                               std::span<const double> shares) {
    using selection::AlgorithmType;
    std::vector<char> partnered(db.size(), 0);
    for (const auto &pair : pairs)
        partnered[pair.i] = partnered[pair.j] = 1;
    std::vector<Arm> arms;
    for (const double share : shares) {
        const auto budget = allocation::total_budget(db, share);
        auto geometric = [&](double alpha) {
            return allocation::create_trajectory_budgets(
                db, allocation::AllocatorType::SizeProportional, budget, alpha);
        };
        const auto equal = geometric(0.0);
        arms.push_back({"RDP_GLOBAL",
                        share,
                        core::to_kept(selection::rdp::simplify_database(db, budget)),
                        {}});
        arms.push_back({"RDP_a0", share, select(db, AlgorithmType::RDP, equal), {}});
        arms.push_back(
            {"RDP_a025", share, select(db, AlgorithmType::RDP, geometric(kRdpAlpha)), {}});
        arms.push_back({"SED_a0", share, select(db, AlgorithmType::SED, equal), {}});
        arms.push_back({"UNI_a0", share, select(db, AlgorithmType::UNIFORM, equal), {}});
        for (const double lambda : kPartnerLambdas) {
            std::vector<double> weights(db.size());
            for (std::size_t i = 0; i < db.size(); ++i)
                weights[i] = 1.0 + (partnered[i] ? lambda : 0.0);
            arms.push_back({std::format("PARTNER_l{:g}", lambda),
                            share,
                            select(db, AlgorithmType::RDP,
                                   weighted_budgets(db, weights, budget - 2 * db.size())),
                            {}});
        }
    }
    return arms;
}

// Per trajectory: a bound on its own Fréchet deviation, and the held-out kNN and join answers it
// lost (in the truth, not answered) or gained falsely. Join counts sum over every join workload
// Held-out kNN: how far each true neighbour's distance moved, as a share of its deviation bound,
// and recall when a false neighbour within (1 + eps) of the true k-th distance counts as a hit
void diagnose_knn(std::ofstream &out, std::uint64_t seed, const Arm &arm,
                  const core::TrajectoryDb &db, const core::TrajectoryDb &compressed,
                  const query::KnnWorkload &workload, std::span<const double> deviation,
                  std::span<const std::vector<std::size_t>> found) {
    constexpr std::array kEps{0.01, 0.05, 0.1};
    const std::size_t m = workload.entries.size();
    std::vector<std::vector<double>> ratios(m);
    std::vector<std::array<double, kEps.size()>> tolerant(m);
    std::vector<double> recall(m, kNan);
    core::parallel_for(m, [&](std::size_t q) {
        const auto &truth = workload.entries[q].reference;
        if (truth.empty())
            return;
        const auto query = workload.queries.trajectory(workload.entries[q].query_index);
        auto distance = [&](const core::TrajectoryView &t) {
            return query::frechet_distance(query, t, workload.frechet);
        };
        for (const auto i : truth)
            if (deviation[i] > 0)
                ratios[q].push_back(
                    (distance(compressed.trajectory(i)) - distance(db.trajectory(i))) /
                    deviation[i]);
        const double kth = distance(db.trajectory(truth.back()));
        std::size_t hits = 0;
        std::array<std::size_t, kEps.size()> near{};
        for (const auto j : found[q]) {
            if (std::ranges::find(truth, j) != truth.end()) {
                ++hits;
                continue;
            }
            const double d = distance(db.trajectory(j));
            for (std::size_t e = 0; e < kEps.size(); ++e)
                near[e] += d <= (1.0 + kEps[e]) * kth;
        }
        const auto k = static_cast<double>(truth.size());
        recall[q] = static_cast<double>(hits) / k;
        for (std::size_t e = 0; e < kEps.size(); ++e)
            tolerant[q][e] = static_cast<double>(hits + near[e]) / k;
    });
    std::vector<double> all;
    for (const auto &r : ratios)
        all.insert(all.end(), r.begin(), r.end());
    std::ranges::sort(all);
    auto quantile = [&](double p) {
        return all.empty() ? kNan : all[static_cast<std::size_t>(p * (all.size() - 1))];
    };
    double mean = 0, positive = 0;
    for (const double r : all) {
        mean += r;
        positive += r > 0;
    }
    if (!all.empty()) {
        mean /= static_cast<double>(all.size());
        positive /= static_cast<double>(all.size());
    }
    double sum = 0;
    std::array<double, kEps.size()> sums{};
    std::size_t count = 0;
    for (std::size_t q = 0; q < m; ++q) {
        if (std::isnan(recall[q]))
            continue;
        ++count;
        sum += recall[q];
        for (std::size_t e = 0; e < kEps.size(); ++e)
            sums[e] += tolerant[q][e];
    }
    const auto c = static_cast<double>(std::max<std::size_t>(count, 1));
    out << std::format("{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},"
                       "{:.4f}\n",
                       seed, arm.share, arm.name, all.size(), quantile(0.1), quantile(0.5),
                       quantile(0.9), mean, positive, sum / c, sums[0] / c, sums[1] / c,
                       sums[2] / c, quantile(0.99));
    out.flush();
}

void diagnose(std::ofstream &out, std::ofstream &knn_out, std::uint64_t seed, const Arm &arm,
              const core::TrajectoryDb &db, const core::TrajectoryDb &compressed,
              const evaluator::Workloads &test) {
    const std::size_t n = db.size();
    std::vector<double> deviation(n);
    core::parallel_for(n, [&](std::size_t i) {
        deviation[i] = teacher::simplification_bound(db.trajectory(i), compressed.trajectory(i));
    });
    std::vector<std::size_t> knn_truth(n), knn_lost(n), knn_false(n);
    if (test.knn) {
        std::unordered_map<std::int32_t, std::size_t> index;
        for (std::size_t i = 0; i < n; ++i)
            index[db.trajectory(i).trajectory_id] = i;
        const auto answers = evaluator::evaluate_knn(compressed, *test.knn).answers;
        std::vector<std::vector<std::size_t>> found(answers.size());
        for (std::size_t q = 0; q < answers.size(); ++q) {
            const auto &truth = test.knn->entries[q].reference;
            for (const auto id : answers[q].neighbors)
                found[q].push_back(index.at(id));
            for (const auto i : truth) {
                ++knn_truth[i];
                knn_lost[i] += std::ranges::find(found[q], i) == found[q].end();
            }
            for (const auto i : found[q])
                knn_false[i] += std::ranges::find(truth, i) == truth.end();
        }
        diagnose_knn(knn_out, seed, arm, db, compressed, *test.knn, deviation, found);
    }
    std::vector<std::size_t> join_truth(n), join_lost(n), join_false(n);
    std::vector<const query::JoinWorkload *> joins;
    if (test.join)
        joins.push_back(test.join);
    for (const auto &workload : test.join_grid)
        if (&workload != test.join)
            joins.push_back(&workload);
    for (const auto *workload : joins) {
        if (workload->mode != query::JoinMode::HeldOut)
            continue;
        std::vector<std::pair<std::size_t, std::size_t>> matched;
        evaluator::evaluate_join(compressed, *workload, &matched);
        std::ranges::sort(matched);
        for (std::size_t e = 0; e < workload->entries.size(); ++e) {
            const auto &entry = workload->entries[e];
            for (const auto i : entry.reference) {
                ++join_truth[i];
                join_lost[i] += !std::ranges::binary_search(
                    matched, std::pair<std::size_t, std::size_t>{entry.query_index, i});
            }
        }
        std::unordered_map<std::size_t, std::size_t> entry_of;
        for (std::size_t e = 0; e < workload->entries.size(); ++e)
            entry_of[workload->entries[e].query_index] = e;
        for (const auto &[q, i] : matched)
            join_false[i] +=
                !std::ranges::binary_search(workload->entries[entry_of.at(q)].reference, i);
    }
    for (std::size_t i = 0; i < n; ++i)
        out << std::format("{},{},{},{},{},{},{:.3f},{},{},{},{},{},{}\n", seed, arm.share,
                           arm.name, i, db.trajectory(i).size(), arm.kept[i].size(), deviation[i],
                           knn_truth[i], knn_lost[i], knn_false[i], join_truth[i], join_lost[i],
                           join_false[i]);
    out.flush();
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

// FNV-1a over every kept index, so two runs can be checked for identical selections
std::uint64_t kept_hash(const KeptSets &kept) {
    std::uint64_t hash = 14695981039346656037ull;
    auto mix = [&](std::uint64_t value) {
        for (unsigned int byte = 0; byte < 8; ++byte) {
            hash ^= value & 0xff;
            hash *= 1099511628211ull;
            value >>= 8;
        }
    };
    for (const auto &trajectory : kept) {
        mix(trajectory.size());
        for (const auto index : trajectory)
            mix(index);
    }
    return hash;
}

// Whether the teacher trained on the join at eps_m in mode: one of its held-out thresholds
bool trained_on(const pipeline::Config &config, double eps_m, query::JoinMode mode) {
    const auto extra = pipeline::teacher_extra_join_eps_m(config);
    return mode == query::JoinMode::HeldOut &&
           (eps_m == config.join.eps_m || std::ranges::find(extra, eps_m) != extra.end());
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
        "norm_clustering,J4,J_T,predicted_clusters,"
        "eval_s,join_mode,self_join_f1,norm_self_join,norm_join_main,"
        "norm_clustering_main,self_knn_recall,norm_self_knn,norm_self_join_mean,J4_self,J_T4");
    auto floors =
        open_csv(args.out / "floors.csv",
                 "seed,knn_recall,range_f1,where_at_avg_hit,join_f1,clustering,self_join_f1,"
                 "self_knn_recall");
    auto thresholds =
        open_csv(args.out / "thresholds.csv",
                 "seed,share,arm,join,eps_m,join_f1,norm_join,recall,teacher_threshold");
    auto terms = open_csv(
        args.out / "terms.csv",
        "seed,share,arm,teacher_knn,teacher_range,teacher_where_at,teacher_join,"
        "norm_knn,norm_range,norm_where_at,norm_join,norm_self_join,teacher_shape,teacher_margin");
    auto budgets = open_csv(args.out / "budgets.csv", "seed,share,arm,points,budget");
    auto timings = open_csv(args.out / "timings.csv", "seed,step,seconds");
    auto hashes = open_csv(args.out / "kept_hashes.csv", "seed,share,arm,hash");
    std::ofstream knn_check;
    if (args.knn_check)
        knn_check = open_csv(args.out / "knn_check.csv",
                             "seed,share,arm,proxy_recall,db_recall,test_recall,norm_proxy,"
                             "norm_db,norm_test");
    std::ofstream diagnostics;
    if (args.diagnose)
        diagnostics = open_csv(args.out / "trajectories.csv",
                               "seed,share,arm,trajectory,points,kept,deviation,knn_truth,"
                               "knn_lost,knn_false,join_truth,join_lost,join_false");
    std::ofstream knn_diagnostics;
    if (args.diagnose)
        knn_diagnostics =
            open_csv(args.out / "knn_diagnostics.csv",
                     "seed,share,arm,pairs,shift_p10,shift_p50,shift_p90,shift_mean,shift_positive,"
                     "recall,recall_eps01,recall_eps05,recall_eps10,shift_p99");

    for (const auto seed : args.seeds.value_or(std::vector{base_config.seed})) {
        auto config = base_config;
        config.seed = seed;
        // Pseudo split: the database is split again and stands in for everything; the real query
        // pool is never read. Its own cache, as the database differs
        const auto outer = split::split(all, config.split, seed);
        std::optional<core::TrajectoryDb> inner;
        if (args.pseudo_split)
            inner = core::subset(all, outer.database);
        const auto &source = inner ? *inner : all;
        const auto parts =
            args.pseudo_split ? split::split(source, config.split, *args.pseudo_split) : outer;
        if (args.pseudo_split)
            config.teacher.cache_dir /= std::format("pseudo_{}", *args.pseudo_split);
        const auto db = core::subset(source, parts.database);
        std::cerr << std::format(
            "seed {}: database {} trajectories, {} points{}\n", seed, db.size(), db.num_points(),
            args.pseudo_split ? std::format(" (pseudo split {})", *args.pseudo_split) : "");

        // Database only
        auto start = std::chrono::steady_clock::now();
        auto workloads = teacher::load_or_build_workloads(
            db, pipeline::teacher_workload_config(config), seed, config.teacher.cache_dir);
        const std::size_t outside = args.knn_outside > 0 ? args.knn_outside
                                    : config.teacher.term_weights[teacher::Objective::Margin] > 0
                                        ? teacher::kMarginOutside
                                        : 0;
        if (outside > 0)
            teacher::add_ranked_neighbours(db, workloads, outside, seed, config.teacher.cache_dir);
        const auto pairs = workloads.positives;
        const teacher::Teacher model(db, std::move(workloads),
                                     pipeline::teacher_objective_config(config), seed);
        timings << std::format("{},teacher_setup,{:.2f}\n", seed, core::seconds_since(start));

        std::vector<Arm> arms;
        for (const double share : config.budget_shares) {
            start = std::chrono::steady_clock::now();
            for (auto &arm : model.run(allocation::total_budget(db, share), args.arms))
                arms.push_back({arm.name, share, core::to_kept(arm.simplification), arm.metrics});
            timings << std::format("{},teacher_{:g},{:.2f}\n", seed, share,
                                   core::seconds_since(start));
        }
        if (args.baselines) {
            for (auto &arm : baseline_arms(db, pairs, config.budget_shares))
                arms.push_back(std::move(arm));
            for (const double share : config.budget_shares)
                for (auto &arm : model.single_recipes(allocation::total_budget(db, share)))
                    arms.push_back(
                        {arm.name, share, core::to_kept(arm.simplification), arm.metrics});
        }
        for (const auto &arm : arms) {
            validate(arm, db);
            hashes << std::format("{},{},{},{:016x}\n", seed, arm.share, arm.name,
                                  kept_hash(arm.kept));
        }
        hashes.flush();

        // Held-out data from here on
        start = std::chrono::steady_clock::now();
        const auto pool = core::subset(source, parts.queries);
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
            // The teacher's own estimate of each metric, next to the test's
            const auto teacher_metrics = arm.metrics ? *arm.metrics : objective.metrics(arm.kept);
            const auto &teacher_floors = objective.floors();
            auto norm = [&](teacher::Objective::Term t) {
                return (teacher_metrics[t] - teacher_floors[t]) / (1.0 - teacher_floors[t]);
            };
            // J_T without the query-free terms: the training counterpart of J4
            const double j_t4 =
                (norm(teacher::Objective::Knn) + norm(teacher::Objective::Range) +
                 norm(teacher::Objective::WhereAt) + norm(teacher::Objective::Join)) /
                4.0;
            results << std::format(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2f},"
                "{},{},{},{},{},{},{},{},{},{}\n",
                seed, arm.share, arm.name, allocation::total_budget(db, arm.share),
                scores.knn.recall_at_k, scores.range_f1, scores.where_at.average_hit,
                scores.join.f1, scores.clustering.score, scores.where_at.within_radius,
                scores.join.micro_f1, scores.knn.kendalls_tau_topk, normalized.knn,
                normalized.range, normalized.where_at, normalized.join, normalized.clustering,
                normalized.combined, objective.total(teacher_metrics),
                scores.clustering.predicted_clusters, core::seconds_since(start),
                mode_name(config.join.mode), scores.self_join.f1, normalized.self_join,
                normalized.join_f1, normalized.clustering_score, scores.self_knn.recall_at_k,
                normalized.self_knn, normalized.self_join_mean, normalized.combined_self, j_t4);

            for (std::size_t k = 0; k < other_joins.size(); ++k) {
                const auto &workload = other_joins[k];
                const auto join_scores = evaluator::evaluate_join(compressed, workload);
                const bool teacher_threshold = trained_on(config, workload.eps_m, workload.mode);
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
                // Built on the self join, which the teacher never trains on
                thresholds << std::format("{},{},{},Clustering,{},{},{},,0\n", seed, arm.share,
                                          arm.name, workload.eps_m, score,
                                          (score - cluster_floor[k]) / (1.0 - cluster_floor[k]));
            }
            terms << std::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n", seed, arm.share,
                                 arm.name, norm(teacher::Objective::Knn),
                                 norm(teacher::Objective::Range), norm(teacher::Objective::WhereAt),
                                 norm(teacher::Objective::Join), normalized.knn, normalized.range,
                                 normalized.where_at, normalized.join, normalized.self_join,
                                 norm(teacher::Objective::Shape), norm(teacher::Objective::Margin));
            if (args.knn_check) {
                const double db_knn = objective.knn_recall(compressed);
                knn_check << std::format("{},{},{},{},{},{},{},{},{}\n", seed, arm.share, arm.name,
                                         teacher_metrics[teacher::Objective::Knn], db_knn,
                                         scores.knn.recall_at_k, norm(teacher::Objective::Knn),
                                         (db_knn - db_knn_floor) / (1.0 - db_knn_floor),
                                         normalized.knn);
                knn_check.flush();
            }
            if (args.diagnose && (teacher::is_arm(arm.name) || arm.name == "RDP_a0" ||
                                  arm.name.starts_with("PARTNER")))
                diagnose(diagnostics, knn_diagnostics, seed, arm, db, compressed, test);
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

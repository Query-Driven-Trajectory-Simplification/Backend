#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/combined.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/io/workload_export.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/route_clustering.hpp"
#include "backend/engine/query/where_at_query.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/split/split.hpp"
#include "backend/engine/teacher/teacher.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <magic_enum/magic_enum.hpp>

#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace pipeline {

namespace {

struct Databases {
    core::TrajectoryDb database;
    core::TrajectoryDb queries;
};

Databases load_split(const Config &config) {
    const core::TrajectoryDb all = io::load_trajectories(config.data_path);
    const auto parts = split::split(all, config.split, config.seed);
    return {.database = core::subset(all, parts.database),
            .queries = core::subset(all, parts.queries)};
}

void print_size(std::string_view name, const core::TrajectoryDb &trajectories) {
    std::cout << name << ": " << trajectories.size() << " trajectories, "
              << trajectories.num_points() << " points\n";
}

std::size_t total_budget(const core::TrajectoryDb &trajectories, double share) {
    const std::size_t endpoints = 2 * trajectories.size();
    const double extra = share * static_cast<double>(trajectories.num_points() - endpoints);
    return endpoints + static_cast<std::size_t>(std::llround(extra));
}

void print_scores(const evaluator::Scoretable &scores) {
    std::cout << "  knn top-k tau: " << scores.knn.kendalls_tau_topk
              << ", recall@k: " << scores.knn.recall_at_k << " (" << scores.knn.query_count
              << " queries)\n";
    std::cout << "  where-at: average hit=" << scores.where_at.average_hit
              << ", within radius=" << scores.where_at.within_radius << '\n';
    std::cout << "  range F1: " << scores.range_f1 << ", recall: " << scores.range_recall << '\n';
    std::cout << "  join F1: " << scores.join.f1 << ", precision: " << scores.join.precision
              << ", recall: " << scores.join.recall << ", micro F1: " << scores.join.micro_f1
              << '\n';
    if (!std::isnan(scores.clustering.score) || scores.clustering.reference_clusters != 0)
        std::cout << "  route clustering: " << scores.clustering.score << " ("
                  << scores.clustering.predicted_clusters << " clusters, "
                  << scores.clustering.predicted_covered << " trajectories)\n";
}

void print_normalized(const evaluator::NormalizedScores &scores) {
    std::cout << "  normalized: knn=" << scores.knn << ", range=" << scores.range
              << ", where-at=" << scores.where_at << ", join=" << scores.join;
    if (!std::isnan(scores.clustering))
        std::cout << ", clustering=" << scores.clustering;
    std::cout << ", combined=" << scores.combined << '\n';
}

core::Simplification endpoints_only(const core::TrajectoryDb &trajectories) {
    core::Simplification simplification;
    for (std::size_t i = 0; i < trajectories.size(); ++i) {
        const std::size_t n = trajectories.trajectory(i).size();
        std::vector<std::size_t> kept;
        if (n > 0)
            kept.push_back(0);
        if (n > 1)
            kept.push_back(n - 1);
        simplification.add_trajectory(kept);
    }
    return simplification;
}

} // namespace

void run(const Config &config) {
    const auto [database, queries] = load_split(config);
    print_size("database", database);
    print_size("query pool", queries);

    const auto knn_workload =
        query::generate_knn_queries(database, queries, config.knn, config.seed);
    std::cout << "knn: " << knn_workload.entries.size() << " queries, k=" << knn_workload.k << '\n';
    const auto where_at = query::generate_where_at_queries(database, config.where_at, config.seed);
    std::cout << "where-at: " << where_at.queries.size() << " probes, "
              << where_at.skipped_trajectories << " trips without eligible time\n";
    const auto range = query::generate_range_queries(database, config.range, config.seed);
    std::cout << "range: " << range.queries.size() << " queries\n";
    const auto join = query::generate_join_queries(database, queries, config.join, config.seed);
    std::size_t join_pairs = 0;
    for (const auto &entry : join.entries)
        join_pairs += entry.reference.size();
    std::cout << "join " << magic_enum::enum_name(join.mode) << ": " << join.entries.size()
              << (join.mode == query::JoinMode::Self ? " database trajectories, eps="
                                                     : " queries, eps=")
              << join.eps_m << " m, " << join_pairs << " matching pairs\n";

    query::RouteClusteringWorkload clustering;
    if (config.clustering.enabled) {
        clustering = query::generate_route_clustering(database, config.clustering, &join);
        std::size_t covered = 0;
        for (const auto &members : clustering.reference.members)
            covered += members.size();
        std::cout << "route clustering: " << clustering.reference.members.size()
                  << " clusters covering " << covered << " trajectories, eps=" << clustering.eps_m
                  << " m, min size " << clustering.min_size << '\n';
    }

    if (config.export_results) {
        io::export_workloads(config.export_dir, database, range.queries, knn_workload);
        std::cout << "workloads written to " << config.export_dir.string() << '\n';
    }

    const evaluator::Workloads workloads{.knn = &knn_workload,
                                         .where_at = where_at.queries,
                                         .range = range.queries,
                                         .join = &join,
                                         .clustering =
                                             config.clustering.enabled ? &clustering : nullptr};

    const auto floor_db = core::materialize(database, endpoints_only(database));
    std::cout << "floor, endpoints only: ";
    print_size("compressed", floor_db);
    const auto floor = evaluator::Evaluator(database, floor_db, workloads, config.where_at_scoring)
                           .evaluate()
                           .scores;
    print_scores(floor);

    // Scores one simplification on the held-out workloads, and exports it under name
    auto evaluate = [&](const std::string &label, const std::string &name,
                        const core::Simplification &simplification) {
        const auto compressed = core::materialize(database, simplification);
        std::cout << label << ": ";
        print_size("compressed", compressed);
        const evaluator::Evaluator evaluator(database, compressed, workloads,
                                             config.where_at_scoring);
        const auto evaluation = evaluator.evaluate();
        print_scores(evaluation.scores);
        print_normalized(evaluator::normalize_scores(evaluation.scores, floor));
        if (config.export_results) {
            io::export_trajectories(config.export_dir, name, compressed);
            io::export_results(config.export_dir, name, evaluation);
        }
    };

    // The teacher sees the database only, with its own workloads (kNN and join from the cache)
    std::optional<teacher::Teacher> teacher_model;
    if (config.teacher.enabled) {
        auto teacher_range = config.range;
        teacher_range.query_count = config.teacher.range_query_count;
        auto teacher_workloads =
            teacher::load_or_build_workloads(database,
                                             {.k = config.knn.k,
                                              .join_eps_m = config.join.eps_m,
                                              .frechet = config.join.frechet,
                                              .range = teacher_range,
                                              .where_at = config.where_at},
                                             config.seed, config.teacher.cache_dir);
        teacher_model.emplace(
            database, std::move(teacher_workloads),
            teacher::ObjectiveConfig{.min_tolerance_m = config.where_at_scoring.min_tolerance_m,
                                     .max_tolerance_m = config.where_at_scoring.max_tolerance_m,
                                     .all_join_pairs = config.teacher.all_join_pairs,
                                     .shape_weight = config.teacher.shape_weight},
            config.seed);
    }

    for (const double share : config.budget_shares) {
        for (const double alpha : config.allocation_alphas) {
            const auto budgets = allocation::create_trajectory_budgets(
                database, allocation::AllocatorType::SizeProportional,
                total_budget(database, share), alpha);

            for (const auto algorithm : config.algorithms) {
                std::string name =
                    std::format("{}_{:g}pct", magic_enum::enum_name(algorithm), share * 100);
                // alpha = 1 keeps the old name, which the Travis frontend expects
                if (alpha != 1.0)
                    name += std::format("_a{:g}", alpha);
                evaluate(std::format("{}% {} alpha={}", share * 100,
                                     magic_enum::enum_name(algorithm), alpha),
                         name, selection::simplify_database(database, algorithm, budgets));
            }
        }

        if (teacher_model) {
            std::cout << "teacher at " << share * 100 << "%\n";
            for (const auto &arm :
                 teacher_model->run(total_budget(database, share), config.teacher.arms))
                evaluate(std::format("{}% {} (J_T {:.4f})", share * 100, arm.name, arm.objective),
                         std::format("{}_{:g}pct", arm.name, share * 100), arm.simplification);
        }
    }
}

} // namespace pipeline

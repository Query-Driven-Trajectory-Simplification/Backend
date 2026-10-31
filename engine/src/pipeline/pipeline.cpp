#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/io/workload_export.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/where_at_query.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/split/split.hpp"

#include <magic_enum/magic_enum.hpp>

#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <string>
#include <string_view>

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
    std::cout << "  where-at: within radius=" << scores.where_at.within_radius
              << ", average hit=" << scores.where_at.average_hit << '\n';
    std::cout << "  range recall: " << scores.range_recall << '\n';
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

    if (config.export_results) {
        io::export_workloads(config.export_dir, database, range.queries, knn_workload);
        std::cout << "workloads written to " << config.export_dir.string() << '\n';
    }

    for (const double share : config.budget_shares) {
        const auto budgets = allocation::create_trajectory_budgets(
            database, allocation::AllocatorType::SizeProportional, total_budget(database, share));

        for (const auto algorithm : config.algorithms) {
            const auto simplification = selection::simplify_database(database, algorithm, budgets);
            const auto compressed = core::materialize(database, simplification);
            std::cout << share * 100 << "% " << magic_enum::enum_name(algorithm) << ": ";
            print_size("compressed", compressed);

            const evaluator::Evaluator evaluator(
                database, compressed,
                {.knn = &knn_workload, .where_at = where_at.queries, .range = range.queries},
                config.where_at_scoring);
            const auto evaluation = evaluator.evaluate();
            print_scores(evaluation.scores);

            if (config.export_results) {
                const std::string name =
                    std::format("{}_{:g}pct", magic_enum::enum_name(algorithm), share * 100);
                io::export_trajectories(config.export_dir, name, compressed);
                io::export_results(config.export_dir, name, evaluation);
            }
        }
    }
}

} // namespace pipeline

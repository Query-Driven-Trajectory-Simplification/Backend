#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/where_at_query.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/split/split.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

namespace pipeline {

namespace {

struct Databases {
    core::TrajectoryDb database;
    core::TrajectoryDb queries;
};

Databases load_split(const Config &config) {
    const core::TrajectoryDb all = io::load_trajectories(config.data_path);
    const auto parts =
        split::split(all, split::SplitType::Random, config.query_percentage, config.split_seed);
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

std::string_view algorithm_name(selection::AlgorithmType algorithm) {
    switch (algorithm) {
    case selection::AlgorithmType::RDP:
        return "RDP";
    case selection::AlgorithmType::SED:
        return "SED";
    }
    return "unknown";
}

} // namespace

void run(const Config &config) {
    const auto [database, queries] = load_split(config);
    print_size("database", database);
    print_size("queries", queries);

    const auto knn_workload = query::generate_knn_queries(database, queries, config.knn);
    const auto where_at = query::generate_where_at_queries(database, config.where_at);
    std::cout << "where-at: " << where_at.queries.size() << " probes, "
              << where_at.skipped_trajectories << " trips without eligible time\n";
    const auto range = query::generate_range_queries(database, config.range);
    std::cout << "range: " << range.queries.size() << " queries\n";

    for (const double share : config.budget_shares) {
        const auto budgets = allocation::create_trajectory_budgets(
            database, allocation::AlgorithmType::kSizeProportional, total_budget(database, share));

        for (const auto algorithm : config.algorithms) {
            const auto simplification = selection::simplify_database(database, algorithm, budgets);
            const auto compressed = core::materialize(database, simplification);
            std::cout << share * 100 << "% " << algorithm_name(algorithm) << ": ";
            print_size("compressed", compressed);

            evaluator::Evaluator evaluator(database, compressed,
                                           {.knn = &knn_workload,
                                            .where_at = where_at.queries,
                                            .range = range.queries},
                                           config.where_at_scoring);
            const auto scores = evaluator.evaluate();
            std::cout << "  knn tau: " << scores.knn.kendalls_tau
                      << ", recall@k: " << scores.knn.recall_at_k << '\n';
            std::cout << "  where-at: within radius=" << scores.where_at.within_radius
                      << ", average hit=" << scores.where_at.average_hit << '\n';
            std::cout << "  range recall: " << scores.range_recall << '\n';
        }
    }
}

} // namespace pipeline

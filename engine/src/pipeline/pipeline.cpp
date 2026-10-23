#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/query/where_at_query.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/split/split.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string_view>

namespace pipeline {

namespace {

constexpr double kQueryPercentage = 0.15;
constexpr std::uint64_t kSplitSeed = 1;

// Hardcoded here for now. Should probably be some flag you give --compress-all, or --comress-0.25
// or whatever
constexpr std::array kBudgetShares{0.0025, 0.005, 0.01, 0.02};
constexpr std::array kAlgorithms{selection::AlgorithmType::RDP, selection::AlgorithmType::SED};

struct Databases {
    core::TrajectoryDb database;
    core::TrajectoryDb queries;
};

Databases load_split(const std::filesystem::path &path) {
    const core::TrajectoryDb all = io::load_trajectories(path);
    const auto parts = split::split(all, split::SplitType::Random, kQueryPercentage, kSplitSeed);
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
    const auto [database, queries] = load_split(config.path);
    print_size("database", database);
    print_size("queries", queries);

    const auto where_at = query::generate_where_at_queries(database);
    std::cout << "where-at: " << where_at.queries.size() << " probes, "
              << where_at.skipped_trajectories << " trips without eligible time\n";

    for (const double share : kBudgetShares) {
        const auto budgets = allocation::create_trajectory_budgets(
            database, allocation::AlgorithmType::kSizeProportional, total_budget(database, share));

        for (const auto algorithm : kAlgorithms) {
            const auto simplification = selection::simplify_database(database, algorithm, budgets);
            const auto compressed = core::materialize(database, simplification);
            std::cout << share * 100 << "% " << algorithm_name(algorithm) << ": ";
            print_size("compressed", compressed);
            evaluator::Evaluator evaluator(database, compressed, {.where_at = where_at.queries});
            const auto scores = evaluator.evaluate();
            std::cout << "  where-at: within radius=" << scores.where_at.within_radius
                      << ", average hit=" << scores.where_at.average_hit << '\n';
        }
    }
}

} // namespace pipeline

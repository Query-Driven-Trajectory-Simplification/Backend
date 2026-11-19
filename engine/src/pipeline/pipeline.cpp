#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/io/workload_export.hpp"
#include "backend/engine/pipeline/scorer.hpp"
#include "backend/engine/pipeline/test_workloads.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/split/split.hpp"
#include "backend/engine/teacher/teacher.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <magic_enum/magic_enum.hpp>

#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

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

// alpha = 1 keeps the old name, which the Travis frontend expects
std::string baseline_name(selection::AlgorithmType algorithm, double share, double alpha) {
    std::string name = std::format("{}_{:g}pct", magic_enum::enum_name(algorithm), share * 100);
    if (alpha != 1.0)
        name += std::format("_a{:g}", alpha);
    return name;
}

// Sees the database only. Null unless enabled
std::unique_ptr<teacher::Teacher> make_teacher(const core::TrajectoryDb &database,
                                               const Config &config) {
    if (!config.teacher.enabled)
        return nullptr;
    return std::make_unique<teacher::Teacher>(
        database,
        teacher::load_or_build_workloads(database, teacher_workload_config(config), config.seed,
                                         config.teacher.cache_dir),
        teacher_objective_config(config), config.seed);
}

} // namespace

void run(const Config &config) {
    const auto [database, queries] = load_split(config);
    print_size("database", database);
    print_size("query pool", queries);

    const auto teacher = make_teacher(database, config);
    // The self-kNN reference comes from the teacher's cache; without the teacher J4-self is NaN
    std::optional<query::KnnWorkload> self_knn;
    if (teacher && config.self_join)
        self_knn = teacher::self_knn_workload(teacher->objective().workloads(), config.knn.p);
    const TestWorkloads workloads(database, queries, config, std::move(self_knn));
    if (config.export_results) {
        io::export_workloads(config.export_dir, database, workloads.range.queries, workloads.knn,
                             workloads.join,
                             workloads.clustering ? &*workloads.clustering : nullptr);
        std::cout << "workloads written to " << config.export_dir.string() << '\n';
    }

    const Scorer scorer(database, workloads, config);

    for (const double share : config.budget_shares) {
        const std::size_t budget = allocation::total_budget(database, share);

        for (const double alpha : config.allocation_alphas) {
            const auto budgets = allocation::create_trajectory_budgets(
                database, allocation::AllocatorType::SizeProportional, budget, alpha);
            for (const auto algorithm : config.algorithms)
                scorer.score(std::format("{}% {} alpha={}", share * 100,
                                         magic_enum::enum_name(algorithm), alpha),
                             baseline_name(algorithm, share, alpha),
                             selection::simplify_database(database, algorithm, budgets));
        }

        if (teacher) {
            std::cout << "teacher at " << share * 100 << "%\n";
            for (const auto &arm : teacher->run(budget, config.teacher.arms))
                scorer.score(
                    std::format("{}% {} (J_T {:.4f})", share * 100, arm.name, arm.objective),
                    std::format("{}_{:g}pct", arm.name, share * 100), arm.simplification);
        }
    }
}

} // namespace pipeline

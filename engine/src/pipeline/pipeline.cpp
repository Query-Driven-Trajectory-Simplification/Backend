#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/io/selection_file.hpp"
#include "backend/engine/io/workload_export.hpp"
#include "backend/engine/pipeline/common.hpp"
#include "backend/engine/pipeline/scorer.hpp"
#include "backend/engine/pipeline/test_workloads.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/teacher/teacher.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <magic_enum/magic_enum.hpp>

#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace pipeline {

namespace {

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

    const TestWorkloads workloads(database, queries, config);
    if (config.export_results) {
        io::export_workloads(config.export_dir, database, workloads.range.queries, workloads.knn,
                             workloads.join,
                             workloads.clustering ? &*workloads.clustering : nullptr);
        std::cout << "workloads written to " << config.export_dir.string() << '\n';
    }

    const Scorer scorer(database, workloads, config);
    const auto teacher = make_teacher(database, config);

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

        for (const auto &name : config.external_algorithms) {
            const auto path =
                config.selection_dir / std::format("{}_{:g}pct.bin", name, share * 100);
            const auto kept = io::read_selection(path, database);
            std::size_t points = 0;
            for (const auto &set : kept)
                points += set.size();
            if (points != budget)
                throw std::runtime_error(
                    std::format("{}: {} points, budget {}", path.string(), points, budget));
            scorer.score(std::format("{}% {}", share * 100, name),
                         std::format("{}_{:g}pct", name, share * 100),
                         core::to_simplification(kept));
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

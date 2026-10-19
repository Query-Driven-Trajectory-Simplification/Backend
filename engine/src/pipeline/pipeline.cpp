#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/selection/selector.hpp"

#include <iostream>

namespace pipeline {

void run(const Config &config) {
    core::TrajectoryDb trajectories = io::load_trajectories(config.path);
    std::cout << trajectories.size() << " trajectories, " << trajectories.num_points()
              << " points\n";
    std::cout << trajectories.trajectory(3).size() << std::endl;

    core::TrajectoryView trajectory = trajectories.trajectory(3);

    auto kept_indices = selection::simplify_trajectory(
        trajectory, 
        selection::AlgorithmType::SED, 
        10
    );

    std::cout << "Simplified down to " << kept_indices.size() << " points. Indices:\n";
    for (std::size_t idx : kept_indices) {
        std::cout << "  index: " << idx 
                  << " -> (x: " << trajectory.x[idx] 
                  << ", y: " << trajectory.y[idx] << ")\n";
    }

    std::size_t total_budget = trajectories.num_points() / 2;
    auto budgets = allocation::create_trajectory_budgets(
        trajectories, allocation::AlgorithmType::kSizeProportional, total_budget);


}

} // namespace pipeline

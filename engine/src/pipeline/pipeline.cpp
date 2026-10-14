#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/allocation/size_proportional.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/io/parquet_loader.hpp"

#include <iostream>

namespace pipeline {

void run(const Config &config) {
    core::TrajectoryDb trajectories = io::load_trajectories(config.path);
    std::cout << trajectories.size() << " trips, " << trajectories.num_points() << " points\n";
    std::cout << trajectories.trajectory(3).size();

    std::int64_t total_budget = trajectories.num_points() / 2;
    auto budgets = allocation::proportional_budget(trajectories, total_budget);
    std::cout << "Budget: " << total_budget << "\n";
    for (size_t i = 0; i < budgets.size(); ++i) {
        std::cout << "Trip " << i << ": " << budgets[i] << "\n";
    }
}

} // namespace pipeline

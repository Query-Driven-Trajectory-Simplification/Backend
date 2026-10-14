#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/io/parquet_loader.hpp"

#include <iostream>

namespace pipeline {

void run(const Config& config) {
    core::TrajectoryDb trajectories = io::load_trajectories(config.path);
    std::cout << trajectories.size() << " trips, " << trajectories.num_points() << " points\n";
    std::cout << trajectories.trip(3).size();
}

}

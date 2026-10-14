#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/core/types.hpp"
#include "backend/engine/io/parquet_loader.hpp"

namespace pipeline {

void run(const Config& config) {
    core::TrajectoryDb trajectories = io::load_trajectories(config.path);
}

}
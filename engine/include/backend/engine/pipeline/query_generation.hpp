#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/pipeline/config.hpp"

#include <filesystem>

namespace pipeline {

void generate_heat_map(const core::TrajectoryDb &trajectories,
                       const std::filesystem::path &output_dir);
void generate_queries(const core::TrajectoryDb &trajectories, const Config &config,
                      const std::filesystem::path &output_dir);

}

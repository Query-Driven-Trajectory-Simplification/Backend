#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/pipeline/config.hpp"

namespace pipeline {

void generate_heat_map(const core::TrajectoryDb &trajectories, const Config &config);
void generate_queries(const core::TrajectoryDb &trajectories, const Config &config);

}

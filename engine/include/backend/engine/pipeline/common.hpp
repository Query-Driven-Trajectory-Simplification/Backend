#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/pipeline/config.hpp"

namespace pipeline {

struct Databases {
    core::TrajectoryDb database;
    core::TrajectoryDb queries;
};

Databases load_split(const Config &config);

} // namespace pipeline

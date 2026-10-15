#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include <filesystem>

namespace io {

core::TrajectoryDb load_trajectories(const std::filesystem::path &path);

}
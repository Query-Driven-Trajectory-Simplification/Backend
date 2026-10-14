#pragma once

#include <filesystem>
#include "backend/engine/core/trajectory_db.hpp"

namespace io {

core::TrajectoryDb load_trajectories(const std::filesystem::path& path);

}
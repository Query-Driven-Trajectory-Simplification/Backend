#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"

#include <filesystem>

namespace io {

// Reads kept points (trajectory_id, point_index) from another tool, e.g. RL4QDTS
core::Simplification load_simplification(const std::filesystem::path &path,
                                         const core::TrajectoryDb &database);

} // namespace io

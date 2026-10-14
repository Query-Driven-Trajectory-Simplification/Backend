#include <arrow/status.h>
#include <filesystem>
#include "backend/engine/core/types.hpp"

namespace io {

core::TrajectoryDb load_trajectories(const std::filesystem::path& path);

}
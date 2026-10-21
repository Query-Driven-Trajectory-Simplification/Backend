#pragma once

#include "backend/engine/core/local_point.hpp"
#include "backend/engine/core/trajectory_db.hpp"

namespace core {

// Position at time t, linearly interpolated in time between the surrounding points.
// t must lie within the trajectory's time span.
core::LocalPoint position_at(const TrajectoryView &trajectory, double t);

} // namespace core


#include "backend/engine/core/trajectory_db.hpp"
#include <cstddef>
namespace selection::sed {

struct DataPoint {
    const std::size_t time;
    const double x;
    const double y;
};
double synchronous_factor(double point_time, double start_time, double end_time);
double synchronous_distance(const core::TrajectoryView &trajectory, std::size_t first,
                            std::size_t last, std::size_t point);

// Optimal minimal maximum SED algorithm
std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                   std::size_t points_to_keep);
} // namespace selection::sed


#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/allocation/size_proportional.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace allocation {

std::vector<std::size_t> create_trajectory_budgets(const core::TrajectoryDb &trajectories,
                                                   AllocatorType allocator,
                                                   std::size_t total_budget, double alpha) {

    switch (allocator) {
    case AllocatorType::SizeProportional:
        return allocation::proportional_budget(trajectories, total_budget, alpha);
    }
    throw std::invalid_argument("create_trajectory_budgets: unknown allocator");
}

std::size_t total_budget(const core::TrajectoryDb &trajectories, double share) {
    const std::size_t endpoints = 2 * trajectories.size();
    const double extra = share * static_cast<double>(trajectories.num_points() - endpoints);
    return endpoints + static_cast<std::size_t>(std::llround(extra));
}
} // namespace allocation

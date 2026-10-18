

#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/allocation/size_proportional.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <cstddef>
#include <stdexcept>

namespace allocation {

std::vector<std::size_t> create_trajectory_budgets(const core::TrajectoryDb &trajectories,
                                                   AlgorithmType algorithm,
                                                   std::size_t total_budget) {

    switch (algorithm) {
    case AlgorithmType::kSizeProportional:
        return allocation::proportional_budget(trajectories, total_budget);
    }
    throw std::invalid_argument("invalid trajectory budget allocator");
}
} // namespace allocation

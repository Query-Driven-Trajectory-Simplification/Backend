#include "backend/engine/core/trajectory_db.hpp"
#include <vector>
namespace allocation {

inline constexpr std::size_t kMinPointsPerTrip = 2;

enum class AlgorithmType { SIZE_PROPORTIONAL };

std::vector<size_t> create_trajectory_budgets(const core::TrajectoryDb &trajectories,
                                              AlgorithmType algorithm, std::size_t total_budget);
} // namespace allocation
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/selection/rdp.hpp"
#include "backend/engine/selection/sed.hpp"

#include <stdexcept>

namespace selection {

std::vector<std::size_t> simplify_trajectory(const core::TrajectoryView &trajectory,
                                             AlgorithmType algorithm, std::size_t points_to_keep) {
    switch (algorithm) {
    case AlgorithmType::RDP:
        return rdp::simplify(trajectory, points_to_keep);
    case AlgorithmType::SED:
        return sed::simplify(trajectory, points_to_keep);
    }
    throw std::invalid_argument("Unknown algo type");
}

core::Simplification simplify_database(const core::TrajectoryDb &trajectories,
                                       AlgorithmType algorithm,
                                       std::span<const std::size_t> budgets) {
    if (budgets.size() != trajectories.size())
        throw std::invalid_argument("simplify_database: need one allocation budget per trajectory");

    core::Simplification simplification;
    for (std::size_t i = 0; i < trajectories.size(); ++i)
        simplification.add_trajectory(
            simplify_trajectory(trajectories.trajectory(i), algorithm, budgets[i]));
    return simplification;
}

} // namespace selection


#include "backend/engine/selection/selector.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/selection/rdp.hpp"
#include "backend/engine/selection/sed.hpp"

#include <stdexcept>
#include <vector>

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

    if (budgets.size() != trajectories.size()) {
        throw std::invalid_argument("simplify_database: need one allocation budget per trajectory");
    }
    if (trajectories.size() == 0) {
        throw std::invalid_argument("simplify_database: no trajectories to be simplified");
    }

    std::vector<std::vector<std::size_t>> results(trajectories.size());
    core::parallel_for(trajectories.size(), [&](std::size_t i) {
        results[i] = simplify_trajectory(trajectories.trajectory(i), algorithm, budgets[i]);
    });

    core::Simplification simplification;
    for (auto &result : results) {
        simplification.add_trajectory(std::move(result));
    }

    return simplification;
}

} // namespace selection

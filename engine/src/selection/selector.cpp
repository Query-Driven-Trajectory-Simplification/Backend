#include "backend/engine/selection/selector.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <stdexcept>


namespace selection {

std::vector<std::int64_t> simplify_trajectory(
    const core::TrajectoryView& trajectory, 
    selection::AlgorithmType algorithm,
    const int32_t points_to_keep
) {
    switch (algorithm) {
        case AlgorithmType::Rdp:
            // return rdp::simplify(trajectory, points_to_keep);
            break;
        case AlgorithmType::Sed:
            // return sed::simplify(trajectory, points_to_keep);
            break;
        default:
            throw std::invalid_argument("Unknown algo type");
    }
    
    throw std::runtime_error("Not implemented yet");
}

} // namespace selection
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/selection/rdp.hpp"

#include <stdexcept>

namespace selection {

std::vector<std::int64_t> simplify_trajectory(const core::TrajectoryView &trajectory,
                                              AlgorithmType algorithm,
                                              std::size_t points_to_keep) {
    switch (algorithm) {
    case AlgorithmType::Rdp:
        return rdp::simplify(trajectory, points_to_keep);
    case AlgorithmType::Sed:
        throw std::runtime_error("Sed not implemented yet");
    }
    throw std::invalid_argument("Unknown algo type");
}

}

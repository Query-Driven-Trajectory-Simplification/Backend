#include "backend/engine/selection/uniform.hpp"

#include <algorithm>

namespace selection::uniform {

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep) {
    const std::size_t n = trajectory.size();
    const std::size_t k = std::min(std::max<std::size_t>(points_to_keep, 2), n);

    std::vector<std::size_t> indices(k);
    for (std::size_t i = 0; i < k; ++i)
        indices[i] = k == 1 ? 0 : i * (n - 1) / (k - 1);
    return indices;
}

} // namespace selection::uniform

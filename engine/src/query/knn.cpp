#include "backend/engine/query/knn.hpp"

#include <stdexcept>

namespace query {

std::vector<std::size_t> knn(const core::TrajectoryDb & /*trajectories*/,
                             const core::TrajectoryView & /*query*/, std::size_t /*k*/) {
    throw std::logic_error("query::knn: distance measure not implemented yet");
}

} // namespace query

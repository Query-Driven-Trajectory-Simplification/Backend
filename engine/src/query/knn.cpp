#include "backend/engine/query/knn.hpp"

#include <stdexcept>

namespace query {

std::vector<std::size_t> knn(const core::TrajectoryDb & /*trajectories*/,
                             const core::TrajectoryView & /*query*/, std::size_t /*k*/) {
    throw std::logic_error("query::knn: distance measure not implemented yet");
}

core::TrajectoryDb generate_knn_queries(const core::TrajectoryDb & /*trajectories*/) {
    throw std::logic_error("query::generate_knn_queries: query generation not implemented yet");
}

} // namespace query

#include "backend/engine/core/trajectory_db.hpp"

namespace split {

struct TrajectoryDbSplit {
    std::vector<size_t> database;
    std::vector<size_t> queries;
};

enum SplitType {
    Random,
    Time
};

TrajectoryDbSplit split(const core::TrajectoryDb& trajectories, SplitType split_type, double query_percentage);

}
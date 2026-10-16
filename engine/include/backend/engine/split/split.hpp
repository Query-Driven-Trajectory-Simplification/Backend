#include "backend/engine/core/trajectory_db.hpp"

namespace split {

struct TrajectoryDbSplit {
    core::TrajectoryDb trainingData;
    core::TrajectoryDb testData;
};

enum SplitType {
    Random,
    Time
};

TrajectoryDbSplit split(core::TrajectoryDb, SplitType);

}
#pragma once

#include <cstdint>
#include <vector>

namespace core {

class TrajectoryDb {
public:
    TrajectoryDb() = default;

    explicit TrajectoryDb(size_t total_points) {
        global_lats.reserve(total_points);
        global_lons.reserve(total_points);
        global_timestamps.reserve(total_points);
    }


private:
    std::vector<double> global_lats;
    std::vector<double> global_lons;
    std::vector<int64_t> global_timestamps;

    std::vector<int32_t> trip_offsets{0};
    std::vector<int32_t> trajectory_ids;
};

}
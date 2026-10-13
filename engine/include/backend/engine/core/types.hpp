#pragma once

#include <cstdint>
#include <vector>

struct TrajectoryDatabase {
    std::vector<double> global_lats;
    std::vector<double> global_lons;
    std::vector<int64_t> global_timestamps;

    std::vector<int64_t> trip_offsets{0};
    std::vector<int64_t> trip_ids;
};

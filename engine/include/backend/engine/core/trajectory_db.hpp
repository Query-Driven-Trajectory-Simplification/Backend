#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace core {

struct TripView {
    std::int32_t trajectory_id;
    std::span<const std::int64_t> t;
    std::span<const double> lat;
    std::span<const double> lon;

    std::size_t size() const { return t.size(); }
};

class TrajectoryDb {
public:
    TrajectoryDb(
                std::vector<std::uint64_t> offsets,
                std::vector<std::int32_t> source_ids,
                std::vector<std::int64_t> t,
                std::vector<double> lat,
                std::vector<double> lon);

    std::size_t size() const { return source_ids.size(); }
    std::size_t num_points() const { return t.size(); }

    TripView trip(std::size_t i) const {
        const auto start = offsets[i];
        const auto len   = offsets[i + 1] - start;
        return TripView{
            source_ids[i],
            {t.data() + start, len},
            {lat.data() + start, len},
            {lon.data() + start, len}
        };
    }

private:
    std::vector<std::uint64_t> offsets;
    std::vector<std::int32_t>  source_ids;
    std::vector<std::int64_t>  t;
    std::vector<double>        lat;
    std::vector<double>        lon;
};

}
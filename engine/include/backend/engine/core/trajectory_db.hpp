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

    std::size_t size() const { return source_ids_.size(); }
    std::size_t num_points() const { return t_.size(); }

    TripView trip(std::size_t i) const {
        const auto start = offsets_[i];
        const auto len   = offsets_[i + 1] - start;
        return TripView{
            source_ids_[i],
            {t_.data() + start, len},
            {lat_.data() + start, len},
            {lon_.data() + start, len}
        };
    }

private:
    std::vector<std::uint64_t> offsets_;
    std::vector<std::int32_t>  source_ids_;
    std::vector<std::int64_t>  t_;
    std::vector<double>        lat_;
    std::vector<double>        lon_;
};

}
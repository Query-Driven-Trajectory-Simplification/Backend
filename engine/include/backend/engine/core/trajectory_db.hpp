#pragma once

#include <cstdint>
#include <vector>

namespace core {

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

private:
    std::vector<std::uint64_t> offsets_;
    std::vector<std::int32_t>  source_ids_;
    std::vector<std::int64_t>  t_;
    std::vector<double>        lat_;
    std::vector<double>        lon_;
};

}
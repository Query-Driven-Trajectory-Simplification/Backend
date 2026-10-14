#include "backend/engine/core/trajectory_db.hpp"
#include <stdexcept>
#include <utility>
namespace core {

TrajectoryDb::TrajectoryDb(
            std::vector<std::uint64_t> offsets,
            std::vector<std::int32_t> source_ids,
            std::vector<std::int64_t> t,
            std::vector<double> lat,
            std::vector<double> lon) : 
            offsets_(std::move(offsets)),
            source_ids_(std::move(source_ids)),
            t_(std::move(t)),
            lat_(std::move(lat)),
            lon_(std::move(lon)) 
{
    if (lat_.size() != t_.size() || lon_.size() != t_.size())
        throw std::invalid_argument("TrajectoryDb: t/lat/lon have different lengths");
    if (offsets_.size() != source_ids_.size() + 1
        || offsets_.front() != 0
        || offsets_.back() != t_.size())
        throw std::invalid_argument("TrajectoryDb: offsets don't match the points");
}

}
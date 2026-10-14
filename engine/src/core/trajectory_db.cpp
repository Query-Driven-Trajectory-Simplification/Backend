#include "backend/engine/core/trajectory_db.hpp"
#include <stdexcept>
namespace core {

TrajectoryDb::TrajectoryDb(
            std::vector<std::uint64_t> offsets,
            std::vector<std::int32_t> source_ids,
            std::vector<std::int64_t> t,
            std::vector<double> lat,
            std::vector<double> lon) : 
            offsets(std::move(offsets)),
            source_ids(std::move(source_ids)),
            t(std::move(t)),
            lat(std::move(lat)),
            lon(std::move(lon)) 
{
    if (lat.size() != t.size() || lon.size() != t.size())
        throw std::invalid_argument("TrajectoryDb: t/lat/lon have different lengths");
    if (offsets.size() != source_ids.size() + 1
        || offsets.front() != 0
        || offsets.back() != t.size())
        throw std::invalid_argument("TrajectoryDb: offsets don't match the points");
}

}
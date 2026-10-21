#pragma once

#include "backend/data/filters/boundingbox.hpp"
#include "backend/engine/core/trajectory_db.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>

namespace query {

inline constexpr double kMinLat = filters::kBeijing.min_lat;
inline constexpr double kMaxLat = filters::kBeijing.max_lat;
inline constexpr double kMinLon = filters::kBeijing.min_lon;
inline constexpr double kMaxLon = filters::kBeijing.max_lon;

inline constexpr double kCellLat = 0.02;
inline constexpr double kCellLon = 0.02;
inline constexpr std::int64_t kCellSeconds = 7 * 24 * 60 * 60;

// Start of cell t = 0, Unix seconds UTC (2007-04-12 01:31:32). Later cells are kCellSeconds apart.
inline constexpr std::int64_t kOriginT = 1176341492;

struct Cell {
    std::int64_t lat;
    std::int64_t lon;
    std::int64_t t;

    auto operator<=>(const Cell &) const = default;
};

struct CellBounds {
    double min_lat;
    double max_lat;
    double min_lon;
    double max_lon;
    std::int64_t t_start;
    std::int64_t t_end;
};

std::optional<Cell> cell_of(double lat, double lon, std::int64_t t);

CellBounds bounds_of(const Cell &cell);

using HeatMap = std::map<Cell, std::uint32_t>;

HeatMap heat_map(const core::TrajectoryDb &db);
HeatMap heat_map(const core::TrajectoryDb &db, std::size_t &dropped_points);

} // namespace query
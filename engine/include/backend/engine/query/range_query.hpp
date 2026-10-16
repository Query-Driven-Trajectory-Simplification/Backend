#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace query {

inline constexpr double kMinLat = 39.4;
inline constexpr double kMaxLat = 41.59471;
inline constexpr double kMinLon = 115.7001;
inline constexpr double kMaxLon = 117.39994;

inline constexpr double kCellLat = 0.02;
inline constexpr double kCellLon = 0.02;
inline constexpr std::int64_t kCellSeconds = 7 * 24 * 60 * 60;

inline constexpr std::int64_t kOriginT = 1176341492;

struct Cell {
    std::int64_t lat;
    std::int64_t lon;
    std::int64_t t;

    auto operator<=>(const Cell &) const = default;
};

struct CellHash {
    std::size_t operator()(const Cell &c) const noexcept;
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

using HeatMap = std::unordered_map<Cell, std::uint32_t, CellHash>;

HeatMap heat_map(const core::TrajectoryDb &db);
HeatMap heat_map(const core::TrajectoryDb &db, std::size_t &dropped_points);

std::vector<std::pair<Cell, HeatMap::mapped_type>> ranked_cells(const HeatMap &heat);

// Only cells with at least this many trajectories can become a query.
inline constexpr std::uint32_t kMinQueryWeight = 2;
inline constexpr std::size_t kQueriesPerSeed = 100;

// A range query is one heat map cell: the box over its lat/lon bounds and its week.
struct RangeQuery {
    Cell cell;
    CellBounds bounds;
    std::uint32_t weight;
};

// Draws count cells with replacement, with probability proportional to their weight, from the
// cells with weight >= min_weight. The same seed gives the same queries on every platform.
std::vector<RangeQuery> sample_queries(const HeatMap &heat, std::uint64_t seed,
                                       std::size_t count = kQueriesPerSeed,
                                       std::uint32_t min_weight = kMinQueryWeight);

} // namespace query
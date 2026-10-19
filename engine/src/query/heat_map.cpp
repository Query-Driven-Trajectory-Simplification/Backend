#include "backend/engine/query/heat_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace query {

static_assert(kCellSeconds > 0);

namespace {

std::int64_t floor_div(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if (a % b != 0 && (a < 0) != (b < 0)) {
        --q;
    }
    return q;
}

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

} // namespace

std::size_t CellHash::operator()(const Cell &c) const noexcept {
    std::uint64_t h = mix(static_cast<std::uint64_t>(c.lat));
    h = mix(h ^ static_cast<std::uint64_t>(c.lon));
    h = mix(h ^ static_cast<std::uint64_t>(c.t));
    return static_cast<std::size_t>(h);
}

std::optional<Cell> cell_of(double lat, double lon, std::int64_t t) {
    if (!(lat >= kMinLat && lat < kMaxLat && lon >= kMinLon && lon < kMaxLon)) {
        return std::nullopt;
    }
    return Cell{
        .lat = static_cast<std::int64_t>(std::floor((lat - kMinLat) / kCellLat)),
        .lon = static_cast<std::int64_t>(std::floor((lon - kMinLon) / kCellLon)),
        .t = floor_div(t - kOriginT, kCellSeconds),
    };
}

CellBounds bounds_of(const Cell &cell) {
    return CellBounds{
        .min_lat = kMinLat + static_cast<double>(cell.lat) * kCellLat,
        .max_lat = kMinLat + static_cast<double>(cell.lat + 1) * kCellLat,
        .min_lon = kMinLon + static_cast<double>(cell.lon) * kCellLon,
        .max_lon = kMinLon + static_cast<double>(cell.lon + 1) * kCellLon,
        .t_start = kOriginT + cell.t * kCellSeconds,
        .t_end = kOriginT + (cell.t + 1) * kCellSeconds,
    };
}

HeatMap heat_map(const core::TrajectoryDb &db, std::size_t &dropped_points) {
    HeatMap heat;
    std::vector<Cell> cells;
    dropped_points = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto &trajectory = db.trajectory(i);

        cells.clear();
        cells.reserve(trajectory.size());
        for (std::size_t k = 1; k + 1 < trajectory.size(); ++k) {
            if (auto cell = cell_of(trajectory.lat[k], trajectory.lon[k], trajectory.t[k])) {
                cells.push_back(*cell);
            } else {
                ++dropped_points;
            }
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());

        for (const auto &cell : cells) {
            ++heat[cell];
        }
    }
    return heat;
}

HeatMap heat_map(const core::TrajectoryDb &db) {
    std::size_t dropped_points = 0;
    return heat_map(db, dropped_points);
}

std::vector<std::pair<Cell, HeatMap::mapped_type>> ranked_cells(const HeatMap &heat) {
    std::vector<std::pair<Cell, HeatMap::mapped_type>> ranked(heat.begin(), heat.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    return ranked;
}

} // namespace query
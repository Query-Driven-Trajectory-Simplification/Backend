#include "backend/engine/io/frontend_export.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace io {

namespace {

using Json = nlohmann::ordered_json;

double round6(double value) { return std::round(value * 1e6) / 1e6; }

Json cell_json(const query::Cell &cell, std::uint32_t weight) {
    const auto b = query::bounds_of(cell);
    return {
        {"lat_idx", cell.lat},
        {"lon_idx", cell.lon},
        {"t_idx", cell.t},
        {"min_lat", round6(b.min_lat)},
        {"max_lat", round6(b.max_lat)},
        {"min_lon", round6(b.min_lon)},
        {"max_lon", round6(b.max_lon)},
        {"t_start", b.t_start},
        {"t_end", b.t_end},
        {"weight", weight},
    };
}

Json range_query_json(const query::RangeQuery &q) {
    return {
        {"trajectory_id", q.trajectory_id},
        {"lat", round6(q.lat)},
        {"lon", round6(q.lon)},
        {"t", q.t},
        {"min_lat", round6(q.min_lat)},
        {"max_lat", round6(q.max_lat)},
        {"min_lon", round6(q.min_lon)},
        {"max_lon", round6(q.max_lon)},
        {"t_start", q.t_start},
        {"t_end", q.t_end},
        {"results", q.results},
    };
}

void write_file(const Json &json, const std::filesystem::path &path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open " + path.string());
    }
    out << json.dump() << '\n';
    if (!out) {
        throw std::runtime_error("failed writing " + path.string());
    }
}

} // namespace

void export_heat_map(const query::HeatMap &heat_map, const std::filesystem::path &path) {
    std::vector<std::pair<query::Cell, std::uint32_t>> cells(heat_map.begin(), heat_map.end());
    std::sort(cells.begin(), cells.end(), [](const auto &a, const auto &b) {
        return std::tie(a.first.t, a.first.lat, a.first.lon) <
               std::tie(b.first.t, b.first.lat, b.first.lon);
    });

    Json json = {
        {"cell_size",
         {{"lat", query::kCellLat}, {"lon", query::kCellLon}, {"seconds", query::kCellSeconds}}},
        {"cells", Json::array()},
    };
    for (const auto &[cell, weight] : cells) {
        json["cells"].push_back(cell_json(cell, weight));
    }
    write_file(json, path);
}

void export_range_queries(std::span<const query::RangeQuery> queries, std::uint64_t seed,
                          const query::RangeQueryParams &params,
                          const std::filesystem::path &path) {
    Json json = {
        {"box_km", params.box_km},
        {"window_seconds", params.window_seconds},
        {"min_results", params.min_results},
        {"full_accept_at", params.full_accept_at},
        {"seed", seed},
        {"queries", Json::array()},
    };
    for (const auto &q : queries) {
        json["queries"].push_back(range_query_json(q));
    }
    write_file(json, path);
}

} // namespace io

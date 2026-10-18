#include "backend/engine/io/frontend_export.hpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace io {

namespace {

void append_cell(std::string &json, const query::Cell &cell, std::uint32_t weight) {
    const auto b = query::bounds_of(cell);
    std::format_to(std::back_inserter(json),
                   "{{\"lat_idx\":{},\"lon_idx\":{},\"t_idx\":{},"
                   "\"min_lat\":{:.6f},\"max_lat\":{:.6f},\"min_lon\":{:.6f},\"max_lon\":{:.6f},"
                   "\"t_start\":{},\"t_end\":{},\"weight\":{}}}",
                   cell.lat, cell.lon, cell.t, b.min_lat, b.max_lat, b.min_lon, b.max_lon,
                   b.t_start, b.t_end, weight);
}

void write_file(const std::string &json, const std::filesystem::path &path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open " + path.string());
    }
    out << json;
    if (!out) {
        throw std::runtime_error("failed writing " + path.string());
    }
}

} // namespace

void export_heat_map(const query::HeatMap &heat_map, const std::filesystem::path &path) {
    // Sorted so the same heat map always gives the same file.
    std::vector<std::pair<query::Cell, std::uint32_t>> cells(heat_map.begin(), heat_map.end());
    std::sort(cells.begin(), cells.end(), [](const auto &a, const auto &b) {
        return std::tie(a.first.t, a.first.lat, a.first.lon) <
               std::tie(b.first.t, b.first.lat, b.first.lon);
    });

    std::string json =
        std::format("{{\"cell_size\":{{\"lat\":{},\"lon\":{},\"seconds\":{}}},\"cells\":[",
                    query::kCellLat, query::kCellLon, query::kCellSeconds);
    for (std::size_t i = 0; i < cells.size(); ++i) {
        json += i == 0 ? "\n" : ",\n";
        append_cell(json, cells[i].first, cells[i].second);
    }
    json += "\n]}\n";
    write_file(json, path);
}

void export_range_queries(std::span<const SeedQueries> seeds, std::uint32_t min_weight,
                          const std::filesystem::path &path) {
    const std::size_t per_seed = seeds.empty() ? 0 : seeds.front().queries.size();
    std::string json = std::format("{{\"min_weight\":{},\"queries_per_seed\":{},\"seeds\":[",
                                   min_weight, per_seed);
    for (std::size_t s = 0; s < seeds.size(); ++s) {
        std::format_to(std::back_inserter(json), "{}\n{{\"seed\":{},\"queries\":[",
                       s == 0 ? "" : ",", seeds[s].seed);
        const auto &queries = seeds[s].queries;
        for (std::size_t i = 0; i < queries.size(); ++i) {
            json += i == 0 ? "\n" : ",\n";
            append_cell(json, queries[i].cell, queries[i].weight);
        }
        json += "\n]}";
    }
    json += "\n]}\n";
    write_file(json, path);
}

void export_box_queries(std::span<const SeedBoxQueries> seeds, const query::BoxQueryParams &params,
                        const std::filesystem::path &path) {
    std::string json = std::format("{{\"box_km\":{},\"window_seconds\":{},\"min_results\":{},"
                                   "\"full_accept_at\":{},\"queries_per_seed\":{},\"seeds\":[",
                                   params.box_km, params.window_seconds, params.min_results,
                                   params.full_accept_at, params.count);
    for (std::size_t s = 0; s < seeds.size(); ++s) {
        std::format_to(std::back_inserter(json), "{}\n{{\"seed\":{},\"queries\":[",
                       s == 0 ? "" : ",", seeds[s].seed);
        const auto &queries = seeds[s].queries;
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const auto &q = queries[i];
            std::format_to(std::back_inserter(json),
                           "{}{{\"trajectory_id\":{},\"lat\":{:.6f},\"lon\":{:.6f},\"t\":{},"
                           "\"min_lat\":{:.6f},\"max_lat\":{:.6f},\"min_lon\":{:.6f},"
                           "\"max_lon\":{:.6f},\"t_start\":{},\"t_end\":{},\"results\":{}}}",
                           i == 0 ? "\n" : ",\n", q.trajectory_id, q.lat, q.lon, q.t, q.min_lat,
                           q.max_lat, q.min_lon, q.max_lon, q.t_start, q.t_end, q.results);
        }
        json += "\n]}";
    }
    json += "\n]}\n";
    write_file(json, path);
}

} // namespace io

#include "backend/engine/io/workload_export.hpp"

#include "backend/engine/core/projection.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <utility>

namespace io {

namespace {

using json = nlohmann::json;

json trajectory_ids(const core::TrajectoryDb &database, std::span<const std::size_t> indices) {
    json ids = json::array();
    for (const auto i : indices)
        ids.push_back(database.trajectory(i).trajectory_id);
    return ids;
}

void write_file(const json &data, const std::filesystem::path &path) {
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot open " + path.string());
    out << data.dump() << '\n';
}

} // namespace

void export_workloads(const std::filesystem::path &dir, const core::TrajectoryDb &database,
                      std::span<const query::RangeQuery> range, const query::KnnWorkload &knn) {
    json range_json = {{"type", "FeatureCollection"}, {"features", json::array()}};
    for (std::size_t n = 0; n < range.size(); ++n) {
        const auto &[box, reference] = range[n];
        const std::pair<double, double> corners[] = {{box.x_min, box.y_min},
                                                     {box.x_max, box.y_min},
                                                     {box.x_max, box.y_max},
                                                     {box.x_min, box.y_max},
                                                     {box.x_min, box.y_min}};
        json ring = json::array();
        for (const auto [x, y] : corners) {
            const auto corner = core::to_lat_lon(x, y);
            ring.push_back({corner.lon, corner.lat});
        }
        range_json["features"].push_back(
            {{"type", "Feature"},
             {"properties",
              {{"query", n},
               {"t_min", box.t_min},
               {"t_max", box.t_max},
               {"ids", trajectory_ids(database, reference)}}},
             {"geometry", {{"type", "Polygon"}, {"coordinates", json::array({ring})}}}});
    }
    write_file(range_json, dir / "range.geojson");

    json knn_json = {{"k", knn.k}, {"queries", json::array()}};
    for (const auto &entry : knn.entries) {
        knn_json["queries"].push_back(
            {{"id", knn.queries.trajectory(entry.query_index).trajectory_id},
             {"neighbours", trajectory_ids(database, entry.reference)}});
    }
    write_file(knn_json, dir / "knn_queries.json");
}

void export_results(const std::filesystem::path &dir, std::string_view name,
                    const core::TrajectoryDb &compressed, std::span<const query::RangeQuery> range,
                    const query::KnnWorkload &knn) {
    json range_json = json::array();
    for (const auto &[box, reference] : range)
        range_json.push_back(trajectory_ids(compressed, query::range(compressed, box)));
    write_file(range_json, dir / ("range_" + std::string(name) + ".json"));

    json knn_json = json::array();
    for (const auto &entry : knn.entries) {
        const auto trajectory = knn.queries.trajectory(entry.query_index);
        knn_json.push_back(
            {{"id", trajectory.trajectory_id},
             {"neighbours",
              trajectory_ids(compressed, query::knn(compressed, trajectory, knn.k))}});
    }
    write_file(knn_json, dir / ("knn_" + std::string(name) + ".json"));
}

} // namespace io

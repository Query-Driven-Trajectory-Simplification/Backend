#include "backend/engine/io/workload_export.hpp"

#include "backend/engine/core/projection.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>

#include <fstream>
#include <stdexcept>
#include <utility>

namespace io {

namespace {

using json = nlohmann::json;

std::vector<std::int32_t> trajectory_ids(const core::TrajectoryDb &database,
                                         std::span<const std::size_t> indices) {
    std::vector<std::int32_t> ids;
    ids.reserve(indices.size());
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
    out.close();
    if (!out)
        throw std::runtime_error("cannot write " + path.string());
}

void check_arrow(const arrow::Status &status) {
    if (!status.ok())
        throw std::runtime_error(status.ToString());
}

template <typename T> T unwrap(arrow::Result<T> result) {
    check_arrow(result.status());
    return std::move(result).ValueOrDie();
}

void write_parquet(const std::shared_ptr<arrow::Table> &table, const std::filesystem::path &path) {
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    const auto out = unwrap(arrow::io::FileOutputStream::Open(path.string()));
    check_arrow(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out));
    check_arrow(out->Close());
}

json knn_json(std::span<const evaluator::KnnAnswer> answers) {
    json rows = json::array();
    for (const auto &answer : answers)
        rows.push_back({{"id", answer.query_trajectory_id}, {"neighbours", answer.neighbors}});
    return rows;
}

void write_knn_parquet(std::span<const evaluator::KnnAnswer> answers,
                       const std::filesystem::path &path) {
    arrow::Int32Builder query_ids, neighbor_ids;
    arrow::Int64Builder ranks;
    for (const auto &answer : answers) {
        for (std::size_t i = 0; i < answer.neighbors.size(); ++i) {
            check_arrow(query_ids.Append(answer.query_trajectory_id));
            check_arrow(ranks.Append(static_cast<std::int64_t>(i) + 1));
            check_arrow(neighbor_ids.Append(answer.neighbors[i]));
        }
    }
    const auto schema = arrow::schema({arrow::field("query_trajectory_id", arrow::int32()),
                                       arrow::field("rank", arrow::int64()),
                                       arrow::field("neighbor_trajectory_id", arrow::int32())});
    write_parquet(arrow::Table::Make(schema, {unwrap(query_ids.Finish()), unwrap(ranks.Finish()),
                                              unwrap(neighbor_ids.Finish())}),
                  path);
}

void write_range_parquet(std::span<const std::vector<std::int32_t>> answers,
                         const std::filesystem::path &path) {
    arrow::Int64Builder query_indices;
    arrow::Int32Builder trajectory_ids;
    for (std::size_t i = 0; i < answers.size(); ++i) {
        for (const auto id : answers[i]) {
            check_arrow(query_indices.Append(static_cast<std::int64_t>(i)));
            check_arrow(trajectory_ids.Append(id));
        }
    }
    const auto schema = arrow::schema({arrow::field("query_index", arrow::int64()),
                                       arrow::field("trajectory_id", arrow::int32())});
    write_parquet(arrow::Table::Make(
                      schema, {unwrap(query_indices.Finish()), unwrap(trajectory_ids.Finish())}),
                  path);
}

void write_trajectory_parquet(const core::TrajectoryDb &database,
                              const std::filesystem::path &path) {
    arrow::Int32Builder trajectory_ids;
    arrow::Int64Builder timestamps;
    arrow::DoubleBuilder latitudes, longitudes;
    for (std::size_t i = 0; i < database.size(); ++i) {
        const auto trajectory = database.trajectory(i);
        for (std::size_t j = 0; j < trajectory.size(); ++j) {
            check_arrow(trajectory_ids.Append(trajectory.trajectory_id));
            check_arrow(timestamps.Append(trajectory.t[j]));
            check_arrow(latitudes.Append(trajectory.lat[j]));
            check_arrow(longitudes.Append(trajectory.lon[j]));
        }
    }
    const auto schema = arrow::schema(
        {arrow::field("trajectory_id", arrow::int32()), arrow::field("t", arrow::int64()),
         arrow::field("lat", arrow::float64()), arrow::field("lon", arrow::float64())});
    write_parquet(arrow::Table::Make(
                      schema, {unwrap(trajectory_ids.Finish()), unwrap(timestamps.Finish()),
                               unwrap(latitudes.Finish()), unwrap(longitudes.Finish())}),
                  path);
}

json trajectories_json(const core::TrajectoryDb &database) {
    json features = json::array();
    for (std::size_t i = 0; i < database.size(); ++i) {
        const auto trajectory = database.trajectory(i);
        json coordinates = json::array();
        for (std::size_t j = 0; j < trajectory.size(); ++j)
            coordinates.push_back({trajectory.lon[j], trajectory.lat[j]});
        features.push_back(
            {{"type", "Feature"},
             {"properties", {{"trajectory_id", trajectory.trajectory_id}}},
             {"geometry", {{"type", "LineString"}, {"coordinates", coordinates}}}});
    }
    return {{"type", "FeatureCollection"}, {"features", features}};
}

} // namespace

void export_workloads(const std::filesystem::path &dir, const core::TrajectoryDb &database,
                      std::span<const query::RangeQuery> range, const query::KnnWorkload &knn) {
    json range_json = {{"type", "FeatureCollection"}, {"features", json::array()}};
    std::vector<std::vector<std::int32_t>> range_answers;
    range_answers.reserve(range.size());
    for (std::size_t n = 0; n < range.size(); ++n) {
        const auto &[box, reference] = range[n];
        range_answers.push_back(trajectory_ids(database, reference));
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
               {"ids", range_answers.back()}}},
             {"geometry", {{"type", "Polygon"}, {"coordinates", json::array({ring})}}}});
    }
    write_file(range_json, dir / "range.geojson");

    write_range_parquet(range_answers, dir / "range_original.parquet");
    std::vector<evaluator::KnnAnswer> answers;
    answers.reserve(knn.entries.size());
    for (const auto &entry : knn.entries)
        answers.push_back({knn.queries.trajectory(entry.query_index).trajectory_id,
                           trajectory_ids(database, entry.reference)});
    write_file({{"k", knn.k}, {"queries", knn_json(answers)}}, dir / "knn_queries.json");
    write_knn_parquet(answers, dir / "knn_original.parquet");
}

void export_results(const std::filesystem::path &dir, std::string_view name,
                    const evaluator::EvaluationResult &evaluation) {
    write_file(evaluation.range, dir / ("range_" + std::string(name) + ".json"));
    write_file(knn_json(evaluation.knn), dir / ("knn_" + std::string(name) + ".json"));
    write_range_parquet(evaluation.range, dir / ("range_" + std::string(name) + ".parquet"));
    write_knn_parquet(evaluation.knn, dir / ("knn_" + std::string(name) + ".parquet"));
}

void export_trajectories(const std::filesystem::path &dir, std::string_view name,
                         const core::TrajectoryDb &database) {
    const auto filename = "trajectories_" + std::string(name);
    write_file(trajectories_json(database), dir / (filename + ".json"));
    write_trajectory_parquet(database, dir / (filename + ".parquet"));
}

} // namespace io

#include "backend/engine/io/workload_export.hpp"

#include "backend/engine/core/projection.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <stdexcept>
#include <utility>

namespace io {

namespace {

std::vector<std::int32_t> trajectory_ids(const core::TrajectoryDb &database,
                                         std::span<const std::size_t> indices) {
    std::vector<std::int32_t> ids;
    ids.reserve(indices.size());
    for (const auto i : indices)
        ids.push_back(database.trajectory(i).trajectory_id);
    return ids;
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
    const auto properties =
        parquet::WriterProperties::Builder().compression(arrow::Compression::ZSTD)->build();
    check_arrow(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out,
                                           parquet::DEFAULT_MAX_ROW_GROUP_LENGTH, properties));
    check_arrow(out->Close());
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

void write_range_queries_parquet(std::span<const query::RangeQuery> range,
                                 const std::filesystem::path &path) {
    arrow::Int64Builder query_indices, t_mins, t_maxs;
    arrow::ListBuilder lons(arrow::default_memory_pool(), std::make_shared<arrow::DoubleBuilder>());
    arrow::ListBuilder lats(arrow::default_memory_pool(), std::make_shared<arrow::DoubleBuilder>());
    auto &lon_values = static_cast<arrow::DoubleBuilder &>(*lons.value_builder());
    auto &lat_values = static_cast<arrow::DoubleBuilder &>(*lats.value_builder());
    for (std::size_t n = 0; n < range.size(); ++n) {
        const auto &box = range[n].box;
        check_arrow(query_indices.Append(static_cast<std::int64_t>(n)));
        check_arrow(t_mins.Append(box.t_min));
        check_arrow(t_maxs.Append(box.t_max));
        check_arrow(lons.Append());
        check_arrow(lats.Append());
        const std::pair<double, double> corners[] = {{box.x_min, box.y_min},
                                                     {box.x_max, box.y_min},
                                                     {box.x_max, box.y_max},
                                                     {box.x_min, box.y_max}};
        for (const auto [x, y] : corners) {
            const auto corner = core::to_lat_lon(x, y);
            check_arrow(lon_values.Append(corner.lon));
            check_arrow(lat_values.Append(corner.lat));
        }
    }
    const auto schema = arrow::schema(
        {arrow::field("query_index", arrow::int64()), arrow::field("t_min", arrow::int64()),
         arrow::field("t_max", arrow::int64()), arrow::field("lon", arrow::list(arrow::float64())),
         arrow::field("lat", arrow::list(arrow::float64()))});
    write_parquet(arrow::Table::Make(schema, {unwrap(query_indices.Finish()),
                                              unwrap(t_mins.Finish()), unwrap(t_maxs.Finish()),
                                              unwrap(lons.Finish()), unwrap(lats.Finish())}),
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
    write_parquet(
        arrow::Table::Make(schema, {unwrap(trajectory_ids.Finish()), unwrap(timestamps.Finish()),
                                    unwrap(latitudes.Finish()), unwrap(longitudes.Finish())}),
        path);
}

void write_join_parquet(std::span<const evaluator::JoinPair> pairs,
                        const std::filesystem::path &path) {
    arrow::Int32Builder firsts, seconds;
    for (const auto &pair : pairs) {
        check_arrow(firsts.Append(pair.first));
        check_arrow(seconds.Append(pair.second));
    }
    const auto schema = arrow::schema({arrow::field("first_trajectory_id", arrow::int32()),
                                       arrow::field("second_trajectory_id", arrow::int32())});
    write_parquet(arrow::Table::Make(schema, {unwrap(firsts.Finish()), unwrap(seconds.Finish())}),
                  path);
}

void write_clusters_parquet(std::span<const std::vector<std::int32_t>> clusters,
                            const std::filesystem::path &path) {
    arrow::Int64Builder cluster_indices;
    arrow::Int32Builder trajectory_ids;
    arrow::BooleanBuilder medoids;
    for (std::size_t c = 0; c < clusters.size(); ++c) {
        for (std::size_t i = 0; i < clusters[c].size(); ++i) {
            check_arrow(cluster_indices.Append(static_cast<std::int64_t>(c)));
            check_arrow(trajectory_ids.Append(clusters[c][i]));
            check_arrow(medoids.Append(i == 0));
        }
    }
    const auto schema = arrow::schema({arrow::field("cluster", arrow::int64()),
                                       arrow::field("trajectory_id", arrow::int32()),
                                       arrow::field("medoid", arrow::boolean())});
    write_parquet(
        arrow::Table::Make(schema, {unwrap(cluster_indices.Finish()),
                                    unwrap(trajectory_ids.Finish()), unwrap(medoids.Finish())}),
        path);
}

} // namespace

void export_workloads(const std::filesystem::path &dir, const core::TrajectoryDb &database,
                      std::span<const query::RangeQuery> range, const query::KnnWorkload &knn,
                      const query::JoinWorkload &join,
                      const query::RouteClusteringWorkload *clustering) {
    std::vector<std::vector<std::int32_t>> range_answers;
    range_answers.reserve(range.size());
    for (const auto &[box, reference] : range)
        range_answers.push_back(trajectory_ids(database, reference));
    write_range_queries_parquet(range, dir / "range_queries.parquet");

    write_range_parquet(range_answers, dir / "range_original.parquet");
    std::vector<evaluator::KnnAnswer> answers;
    answers.reserve(knn.entries.size());
    for (const auto &entry : knn.entries)
        answers.push_back({knn.queries.trajectory(entry.query_index).trajectory_id,
                           trajectory_ids(database, entry.reference)});
    write_knn_parquet(answers, dir / "knn_original.parquet");

    const bool self = join.mode == query::JoinMode::Self;
    std::vector<evaluator::JoinPair> pairs;
    for (const auto &entry : join.entries) {
        const auto first = self ? database.trajectory(entry.query_index).trajectory_id
                                : join.queries.trajectory(entry.query_index).trajectory_id;
        for (const auto j : entry.reference)
            pairs.push_back({first, database.trajectory(j).trajectory_id});
    }
    write_join_parquet(pairs, dir / "join_original.parquet");

    if (clustering) {
        std::vector<std::vector<std::int32_t>> clusters;
        clusters.reserve(clustering->reference.members.size());
        for (const auto &members : clustering->reference.members)
            clusters.push_back(trajectory_ids(database, members));
        write_clusters_parquet(clusters, dir / "clustering_original.parquet");
    }
}

void export_results(const std::filesystem::path &dir, std::string_view name,
                    const evaluator::EvaluationResult &evaluation, bool clustering) {
    write_range_parquet(evaluation.range, dir / ("range_" + std::string(name) + ".parquet"));
    write_knn_parquet(evaluation.knn, dir / ("knn_" + std::string(name) + ".parquet"));
    write_join_parquet(evaluation.join, dir / ("join_" + std::string(name) + ".parquet"));
    if (clustering)
        write_clusters_parquet(evaluation.clusters,
                               dir / ("clustering_" + std::string(name) + ".parquet"));
}

void export_trajectories(const std::filesystem::path &dir, std::string_view name,
                         const core::TrajectoryDb &database) {
    const auto filename = "trajectories_" + std::string(name);
    write_trajectory_parquet(database, dir / (filename + ".parquet"));
}

} // namespace io

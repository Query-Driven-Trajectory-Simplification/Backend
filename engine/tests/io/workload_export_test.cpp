#include "backend/engine/io/workload_export.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

// Ids 10 (3 points) and 20 (7 points)
core::TrajectoryDb make_database() {
    std::vector<std::int64_t> t;
    std::vector<double> lat, lon;
    for (int i = 0; i < 10; ++i) {
        t.push_back(i);
        lat.push_back(39.9 + i * 0.001);
        lon.push_back(116.3 + i * 0.001);
    }
    return core::TrajectoryDb({0, 3, 10}, {10, 20}, std::move(t), std::move(lat), std::move(lon));
}

std::shared_ptr<arrow::Table> read_table(const std::filesystem::path &path) {
    auto in = arrow::io::ReadableFile::Open(path.string()).ValueOrDie();
    parquet::arrow::FileReaderBuilder builder;
    EXPECT_TRUE(builder.Open(in).ok());
    auto reader = builder.Build().ValueOrDie();
    return reader->ReadTable().ValueOrDie()->CombineChunks().ValueOrDie();
}

template <typename Array>
std::shared_ptr<Array> column(const arrow::Table &table, const char *name) {
    return std::static_pointer_cast<Array>(table.GetColumnByName(name)->chunk(0));
}

TEST(WorkloadExport, RangeBoxesKeepBoxesAndReferenceIds) {
    const auto database = make_database();
    const std::vector<query::RangeQuery> queries{
        {{.x_min = -1.5, .x_max = 2.5, .y_min = -3.5, .y_max = 4.5, .t_min = 100, .t_max = 200},
         {0, 1}},
        {{.x_min = 10.0, .x_max = 20.0, .y_min = 30.0, .y_max = 40.0, .t_min = -5, .t_max = 5},
         {1}},
    };
    const auto path =
        std::filesystem::temp_directory_path() /
        ("backend-range-boxes-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".parquet");

    io::export_range_boxes(path, database, queries);
    const auto table = read_table(path);
    std::filesystem::remove(path);

    ASSERT_EQ(table->num_rows(), 2);
    const auto x_min = column<arrow::DoubleArray>(*table, "x_min");
    const auto x_max = column<arrow::DoubleArray>(*table, "x_max");
    const auto y_min = column<arrow::DoubleArray>(*table, "y_min");
    const auto y_max = column<arrow::DoubleArray>(*table, "y_max");
    const auto t_min = column<arrow::Int64Array>(*table, "t_min");
    const auto t_max = column<arrow::Int64Array>(*table, "t_max");
    for (std::int64_t i = 0; i < 2; ++i) {
        const auto &box = queries[static_cast<std::size_t>(i)].box;
        EXPECT_EQ(x_min->Value(i), box.x_min);
        EXPECT_EQ(x_max->Value(i), box.x_max);
        EXPECT_EQ(y_min->Value(i), box.y_min);
        EXPECT_EQ(y_max->Value(i), box.y_max);
        EXPECT_EQ(t_min->Value(i), box.t_min);
        EXPECT_EQ(t_max->Value(i), box.t_max);
    }

    // References are written as trajectory ids, not database indices
    const auto reference = column<arrow::ListArray>(*table, "reference");
    const auto ids = std::static_pointer_cast<arrow::Int32Array>(reference->values());
    ASSERT_EQ(reference->value_length(0), 2);
    EXPECT_EQ(ids->Value(reference->value_offset(0)), 10);
    EXPECT_EQ(ids->Value(reference->value_offset(0) + 1), 20);
    ASSERT_EQ(reference->value_length(1), 1);
    EXPECT_EQ(ids->Value(reference->value_offset(1)), 20);
}

} // namespace

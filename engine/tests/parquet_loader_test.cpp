#include "backend/engine/io/parquet_loader.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace {
class ParquetLoader : public testing::Test {
  protected:
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("backend-loader-" +
         std::string(testing::UnitTest::GetInstance()->current_test_info()->name()) + "-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".parquet");

    void TearDown() override { std::filesystem::remove(path); }

    void write(bool null_time = false, bool descending_ids = false) {
        arrow::Int32Builder ids;
        arrow::Int64Builder times;
        arrow::DoubleBuilder lat, lon;
        for (int i = 0; i < 6; ++i) {
            ASSERT_TRUE(ids.Append(descending_ids ? (i < 3 ? 20 : 10) : (i < 3 ? 10 : 20)).ok());
            ASSERT_TRUE((null_time && i == 1 ? times.AppendNull() : times.Append(i % 3)).ok());
            ASSERT_TRUE(lat.Append(39.95 + i * 0.001).ok());
            ASSERT_TRUE(lon.Append(116.35 + i * 0.001).ok());
        }
        auto schema = arrow::schema(
            {arrow::field("trajectory_id", arrow::int32()), arrow::field("t", arrow::int64()),
             arrow::field("lat", arrow::float64()), arrow::field("lon", arrow::float64())});
        auto table =
            arrow::Table::Make(schema, {ids.Finish().ValueOrDie(), times.Finish().ValueOrDie(),
                                        lat.Finish().ValueOrDie(), lon.Finish().ValueOrDie()});
        const auto output = arrow::io::FileOutputStream::Open(path.string()).ValueOrDie();
        // Two rows per row group makes each trajectory span row groups.
        ASSERT_TRUE(
            parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output, 2).ok());
        ASSERT_TRUE(output->Close().ok());
    }
};
} // namespace

TEST_F(ParquetLoader, PreservesTrajectoriesAcrossRowGroups) {
    write();
    const auto db = io::load_trajectories(path);
    ASSERT_EQ(db.size(), 2u);
    ASSERT_EQ(db.num_points(), 6u);
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto trip = db.trajectory(i);
        EXPECT_EQ(trip.trajectory_id, i == 0 ? 10 : 20);
        ASSERT_EQ(trip.size(), 3u);
        for (std::size_t j = 0; j < trip.size(); ++j) {
            EXPECT_EQ(trip.t[j], j);
            EXPECT_DOUBLE_EQ(trip.lat[j], 39.95 + (3 * i + j) * 0.001);
            EXPECT_DOUBLE_EQ(trip.lon[j], 116.35 + (3 * i + j) * 0.001);
        }
    }
}

TEST_F(ParquetLoader, RejectsNullPointData) {
    write(true);
    EXPECT_THROW(io::load_trajectories(path), std::runtime_error);
}

TEST_F(ParquetLoader, RejectsDescendingTrajectoryIdsAcrossRowGroups) {
    write(false, true);
    EXPECT_THROW(io::load_trajectories(path), std::runtime_error);
}

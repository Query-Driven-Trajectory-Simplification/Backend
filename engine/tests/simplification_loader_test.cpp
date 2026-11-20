#include "backend/engine/io/simplification_loader.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
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

std::vector<std::size_t> kept(const core::Simplification &s, std::size_t i) {
    const auto span = s.trajectory(i);
    return {span.begin(), span.end()};
}

class SimplificationLoader : public testing::Test {
  protected:
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("backend-kept-" +
         std::string(testing::UnitTest::GetInstance()->current_test_info()->name()) + "-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".parquet");
    core::TrajectoryDb database = make_database();

    void TearDown() override { std::filesystem::remove(path); }

    void write(const std::vector<std::pair<std::int32_t, std::int64_t>> &rows,
               bool int32_indices = false) {
        arrow::Int32Builder ids, small_indices;
        arrow::Int64Builder indices;
        for (const auto &[id, index] : rows) {
            ASSERT_TRUE(ids.Append(id).ok());
            ASSERT_TRUE((int32_indices ? small_indices.Append(static_cast<std::int32_t>(index))
                                       : indices.Append(index))
                            .ok());
        }
        const auto index_type = int32_indices ? arrow::int32() : arrow::int64();
        const auto schema = arrow::schema({arrow::field("trajectory_id", arrow::int32()),
                                           arrow::field("point_index", index_type)});
        const auto index_array =
            int32_indices ? small_indices.Finish().ValueOrDie() : indices.Finish().ValueOrDie();
        const auto table = arrow::Table::Make(schema, {ids.Finish().ValueOrDie(), index_array});
        const auto output = arrow::io::FileOutputStream::Open(path.string()).ValueOrDie();
        ASSERT_TRUE(
            parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output, 4).ok());
        ASSERT_TRUE(output->Close().ok());
    }
};

} // namespace

TEST_F(SimplificationLoader, ReturnsDatabaseOrderSortedWithoutDuplicates) {
    write({{20, 6}, {10, 2}, {20, 0}, {10, 0}, {20, 3}, {10, 1}, {20, 6}});
    const auto simplification = io::load_simplification(path, database);
    ASSERT_EQ(simplification.size(), 2u);
    EXPECT_EQ(simplification.num_points(), 6u);
    EXPECT_EQ(kept(simplification, 0), (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_EQ(kept(simplification, 1), (std::vector<std::size_t>{0, 3, 6}));
}

TEST_F(SimplificationLoader, RejectsAMissingTrajectory) {
    write({{10, 0}, {10, 2}});
    EXPECT_THROW(io::load_simplification(path, database), std::runtime_error);
}

TEST_F(SimplificationLoader, RejectsATrajectoryNotInTheDatabase) {
    write({{10, 0}, {10, 2}, {20, 0}, {20, 6}, {30, 0}});
    EXPECT_THROW(io::load_simplification(path, database), std::runtime_error);
}

TEST_F(SimplificationLoader, RejectsAnIndexOutOfRange) {
    write({{10, 0}, {10, 3}, {20, 0}, {20, 6}});
    EXPECT_THROW(io::load_simplification(path, database), std::runtime_error);
}

TEST_F(SimplificationLoader, RejectsANegativeIndex) {
    write({{10, -1}, {10, 0}, {10, 2}, {20, 0}, {20, 6}});
    EXPECT_THROW(io::load_simplification(path, database), std::runtime_error);
}

TEST_F(SimplificationLoader, RejectsAMissingEndpoint) {
    write({{10, 0}, {10, 2}, {20, 0}, {20, 3}});
    EXPECT_THROW(io::load_simplification(path, database), std::runtime_error);
}

TEST_F(SimplificationLoader, RejectsWrongColumnTypes) {
    write({{10, 0}, {10, 2}, {20, 0}, {20, 6}}, true);
    EXPECT_THROW(io::load_simplification(path, database), std::runtime_error);
}

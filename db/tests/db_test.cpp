#include <cstdint>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "db/db.hpp"

namespace fs = std::filesystem;

namespace {

class DatabaseTest : public ::testing::Test {
protected:
    db::Database db = *db::Database::open("");  // in-memory
    fs::path parquet = fs::temp_directory_path() / "database_test_points.parquet";

    void SetUp() override {
        auto schema = db::create_schema(db);
        ASSERT_TRUE(schema) << schema.error();

        // Same columns as `convert` writes. Trajectory 7 has a duplicate timestamp,
        // so seq_no must follow the file order, not the time.
        auto written = db.query(
            "COPY (SELECT * FROM (VALUES "
            "  (7::INTEGER, 1224730384::BIGINT, 39.98, 116.31, 150.0), "
            "  (7,          1224730384,         39.99, 116.32, NULL), "
            "  (7,          1224730390,         40.00, 116.33, 151.0), "
            "  (3,          1224730000,         39.90, 116.40, NULL)  "
            ") AS v(trajectory_id, t, lat, lon, alt_m)) "
            "TO '" + parquet.string() + "' (FORMAT parquet)");
        ASSERT_TRUE(written) << written.error();
    }

    void TearDown() override { fs::remove(parquet); }

    std::int64_t count(const std::string& sql) {
        auto result = db.query(sql);
        EXPECT_TRUE(result) << result.error();
        return result ? result->int64(0, 0) : -1;
    }
};

}

TEST_F(DatabaseTest, CreateSchemaTwiceIsFine) {
    auto again = db::create_schema(db);
    EXPECT_TRUE(again) << again.error();
}

TEST_F(DatabaseTest, OpenReportsErrors) {
    auto opened = db::Database::open("/does/not/exist/p7.duckdb");
    EXPECT_FALSE(opened);
}

TEST_F(DatabaseTest, LoadsPointsInFileOrder) {
    auto loaded = db::load_points(db, parquet, "geolife");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(*loaded, 4);

    auto r = db.query(
        "SELECT seq_no, lat, epoch(ts)::BIGINT, alt_m "
        "FROM points WHERE trajectory_id = 7 ORDER BY seq_no");
    ASSERT_TRUE(r) << r.error();
    ASSERT_EQ(r->rows(), 3);
    EXPECT_EQ(r->int64(0, 0), 0);
    EXPECT_DOUBLE_EQ(r->number(1, 0), 39.98);
    EXPECT_DOUBLE_EQ(r->number(1, 1), 39.99);  // same timestamp, kept in file order
    EXPECT_EQ(r->int64(2, 0), 1224730384);
    EXPECT_TRUE(r->is_null(3, 1));
}

TEST_F(DatabaseTest, FillsTrajectories) {
    ASSERT_TRUE(db::load_points(db, parquet, "geolife"));
    EXPECT_EQ(count("SELECT num_points FROM trajectories WHERE trajectory_id = 7"), 3);
    EXPECT_EQ(count("SELECT num_points FROM trajectories WHERE trajectory_id = 3"), 1);
}

TEST_F(DatabaseTest, LoadingTwiceReplaces) {
    ASSERT_TRUE(db::load_points(db, parquet, "geolife"));
    ASSERT_TRUE(db::load_points(db, parquet, "geolife"));
    EXPECT_EQ(count("SELECT count(*) FROM points"), 4);
    EXPECT_EQ(count("SELECT count(*) FROM trajectories"), 2);
}

TEST_F(DatabaseTest, MissingFileIsAnErrorAndChangesNothing) {
    ASSERT_TRUE(db::load_points(db, parquet, "geolife"));
    auto loaded = db::load_points(db, "/does/not/exist.parquet", "geolife");
    EXPECT_FALSE(loaded);
    EXPECT_EQ(count("SELECT count(*) FROM points"), 4);  // rolled back
}

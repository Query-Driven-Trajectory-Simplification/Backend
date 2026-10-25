#include "backend/engine/pipeline/config.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::filesystem::path write_config(const std::string &name, const std::string &contents) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << contents;
    return path;
}

} // namespace

TEST(Config, MissingKeysKeepDefaults) {
    const auto config =
        pipeline::load_config(write_config("defaults.json", R"({"data_path": "data.parquet"})"));
    const pipeline::Config defaults;
    EXPECT_EQ(config.data_path, "data.parquet");
    EXPECT_DOUBLE_EQ(config.query_percentage, defaults.query_percentage);
    EXPECT_EQ(config.split_seed, defaults.split_seed);
    EXPECT_EQ(config.budget_shares, defaults.budget_shares);
    EXPECT_EQ(config.algorithms, defaults.algorithms);
    EXPECT_EQ(config.knn.k, defaults.knn.k);
    EXPECT_EQ(config.range.query_count, defaults.range.query_count);
    EXPECT_EQ(config.where_at.probes_per_trajectory, defaults.where_at.probes_per_trajectory);
    EXPECT_DOUBLE_EQ(config.where_at_scoring.hit_radius_m, defaults.where_at_scoring.hit_radius_m);
}

TEST(Config, ReadsTopLevelAndNestedKeys) {
    const auto config = pipeline::load_config(write_config("overrides.json", R"({
        "data_path": "data.parquet",
        "query_percentage": 0.3,
        "split_seed": 7,
        "budget_shares": [0.5],
        "algorithms": ["SED"],
        "knn": {"k": 3, "query_count": 0},
        "range": {"query_count": 10, "box_size_m": 500, "window_seconds": 3600},
        "where_at": {"probes_per_trajectory": 16, "max_gap_seconds": 30},
        "where_at_scoring": {"hit_radius_m": 50, "max_tolerance_m": 300}
    })"));
    EXPECT_DOUBLE_EQ(config.query_percentage, 0.3);
    EXPECT_EQ(config.split_seed, 7);
    EXPECT_EQ(config.budget_shares, std::vector<double>{0.5});
    EXPECT_EQ(config.algorithms, std::vector{selection::AlgorithmType::SED});
    EXPECT_EQ(config.knn.k, 3);
    EXPECT_EQ(config.knn.query_count, 0);
    EXPECT_EQ(config.knn.seed, query::KnnConfig{}.seed);
    EXPECT_EQ(config.range.query_count, 10);
    EXPECT_DOUBLE_EQ(config.range.box_size_m, 500);
    EXPECT_EQ(config.range.window_seconds, 3600);
    EXPECT_EQ(config.range.seed, query::RangeConfig{}.seed);
    EXPECT_EQ(config.where_at.probes_per_trajectory, 16);
    EXPECT_DOUBLE_EQ(config.where_at.max_gap_seconds, 30);
    EXPECT_EQ(config.where_at.seed, query::WhereAtConfig{}.seed);
    EXPECT_DOUBLE_EQ(config.where_at_scoring.hit_radius_m, 50);
    EXPECT_DOUBLE_EQ(config.where_at_scoring.min_tolerance_m,
                     evaluator::WhereAtScoringConfig{}.min_tolerance_m);
    EXPECT_DOUBLE_EQ(config.where_at_scoring.max_tolerance_m, 300);
}

TEST(Config, RejectsMissingDataPathAndWrongTypes) {
    EXPECT_THROW(pipeline::load_config(write_config("no_path.json", "{}")), std::invalid_argument);
    EXPECT_THROW(pipeline::load_config(
                     write_config("bad_type.json", R"({"data_path": "d", "split_seed": "one"})")),
                 std::invalid_argument);
    EXPECT_THROW(pipeline::load_config(write_config(
                     "bad_nested_type.json", R"({"data_path": "d", "knn": {"k": "five"}})")),
                 std::invalid_argument);
    EXPECT_THROW(pipeline::load_config(write_config(
                     "bad_algorithm.json", R"({"data_path": "d", "algorithms": ["DP"]})")),
                 std::invalid_argument);
    EXPECT_THROW(pipeline::load_config("missing.json"), std::invalid_argument);
}

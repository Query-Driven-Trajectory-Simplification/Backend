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
    EXPECT_EQ(config.split.type, defaults.split.type);
    EXPECT_DOUBLE_EQ(config.split.query_percentage, defaults.split.query_percentage);
    EXPECT_EQ(config.seed, 42u);
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
        "seed": 123,
        "split": {"type": "Time", "query_percentage": 0.3},
        "budget_shares": [0.5],
        "algorithms": ["SED"],
        "knn": {"k": 3, "query_count": 0},
        "range": {"query_count": 10, "box_size_m": 500, "window_seconds": 3600},
        "where_at": {"probes_per_trajectory": 16, "max_gap_seconds": 30},
        "where_at_scoring": {"hit_radius_m": 50, "max_tolerance_m": 300}
    })"));
    EXPECT_EQ(config.split.type, split::SplitType::Time);
    EXPECT_DOUBLE_EQ(config.split.query_percentage, 0.3);
    EXPECT_EQ(config.seed, 123u);
    EXPECT_EQ(config.budget_shares, std::vector<double>{0.5});
    EXPECT_EQ(config.algorithms, std::vector{selection::AlgorithmType::SED});
    EXPECT_EQ(config.knn.k, 3);
    EXPECT_EQ(config.knn.query_count, 0);
    EXPECT_EQ(config.range.query_count, 10);
    EXPECT_DOUBLE_EQ(config.range.box_size_m, 500);
    EXPECT_EQ(config.range.window_seconds, 3600);
    EXPECT_EQ(config.where_at.probes_per_trajectory, 16);
    EXPECT_DOUBLE_EQ(config.where_at.max_gap_seconds, 30);
    EXPECT_DOUBLE_EQ(config.where_at_scoring.hit_radius_m, 50);
    EXPECT_DOUBLE_EQ(config.where_at_scoring.min_tolerance_m,
                     evaluator::WhereAtScoringConfig{}.min_tolerance_m);
    EXPECT_DOUBLE_EQ(config.where_at_scoring.max_tolerance_m, 300);
}

TEST(Config, RejectsMissingDataPathAndWrongTypes) {
    EXPECT_THROW(pipeline::load_config(write_config("no_path.json", "{}")), std::invalid_argument);
    EXPECT_THROW(pipeline::load_config(write_config(
                     "bad_type.json", R"({"data_path": "d", "budget_shares": "one"})")),
                 std::invalid_argument);
    EXPECT_THROW(pipeline::load_config(write_config("bad_nested_type.json",
                                                    R"({"data_path": "d", "knn": {"k": "five"}})")),
                 std::invalid_argument);
    EXPECT_THROW(pipeline::load_config(write_config("bad_algorithm.json",
                                                    R"({"data_path": "d", "algorithms": ["DP"]})")),
                 std::invalid_argument);
    EXPECT_THROW(pipeline::load_config(write_config(
                     "bad_split.json", R"({"data_path": "d", "split": {"type": "Space"}})")),
                 std::invalid_argument);
    EXPECT_THROW(pipeline::load_config("missing.json"), std::invalid_argument);
}

TEST(Config, AllowsComments) {
    const auto config = pipeline::load_config(write_config("comments.json", R"({
        // the dataset
        "data_path": "data.parquet",
        "knn": {"k": 3} /* fewer neighbours */
    })"));
    EXPECT_EQ(config.knn.k, 3);
}

TEST(Config, BudgetSharesIncludeBothBoundaries) {
    const auto config = pipeline::load_config(write_config(
        "budget_boundaries.json", R"({"data_path": "d", "budget_shares": [0, 0.5, 1]})"));
    EXPECT_EQ(config.budget_shares, (std::vector<double>{0.0, 0.5, 1.0}));
}

TEST(Config, RejectsInvalidBudgetShares) {
    for (const auto &share : {"-0.01", "1.01", "1e999", "-1e999", "null", "\"NaN\""}) {
        SCOPED_TRACE(share);
        const auto contents =
            std::string(R"({"data_path": "d", "budget_shares": [0.5, )") + share + "]}";
        EXPECT_THROW(pipeline::load_config(write_config("invalid_budget.json", contents)),
                     std::invalid_argument);
    }
}

TEST(Config, RejectsUnknownKeysAndNestedSeeds) {
    EXPECT_THROW(
        pipeline::load_config(write_config("unknown_key.json", R"({"data_path": "d", "sead": 7})")),
        std::invalid_argument);
    for (const auto &section : {"split", "knn", "range", "where_at"}) {
        const auto contents =
            std::string(R"({"data_path": "d", ")") + section + R"(": {"seed": 7}})";
        EXPECT_THROW(pipeline::load_config(write_config("nested_seed.json", contents)),
                     std::invalid_argument);
    }
}

TEST(Config, ExportIsExplicitlyEnabled) {
    EXPECT_FALSE(pipeline::load_config(
                     write_config("export_default.json", R"({"data_path":"data.parquet"})"))
                     .export_results);
    EXPECT_TRUE(
        pipeline::load_config(write_config("export_enabled.json",
                                           R"({"data_path":"data.parquet","export_results":true})"))
            .export_results);
    EXPECT_THROW(
        pipeline::load_config(write_config(
            "export_invalid.json", R"({"data_path":"data.parquet","export_results":"yes"})")),
        std::invalid_argument);
}

TEST(Config, ExportDirectoryDoesNotEnableExport) {
    const auto defaults = pipeline::load_config(
        write_config("export_path_default.json", R"({"data_path":"data.parquet"})"));
    EXPECT_EQ(defaults.export_dir, "output/queries");
    const auto custom = pipeline::load_config(write_config(
        "export_path_custom.json", R"({"data_path":"data.parquet","export_dir":"output/custom"})"));
    EXPECT_EQ(custom.export_dir, "output/custom");
    EXPECT_FALSE(custom.export_results);
    EXPECT_THROW(pipeline::load_config(write_config(
                     "export_path_empty.json",
                     R"({"data_path":"data.parquet","export_results":true,"export_dir":""})")),
                 std::invalid_argument);
}

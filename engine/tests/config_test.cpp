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
    EXPECT_EQ(config.allocation_alphas, defaults.allocation_alphas);
    EXPECT_EQ(config.algorithms, defaults.algorithms);
    EXPECT_EQ(config.knn.k, defaults.knn.k);
    EXPECT_EQ(config.range.query_count, defaults.range.query_count);
    EXPECT_EQ(config.where_at.probes_per_trajectory, defaults.where_at.probes_per_trajectory);
    EXPECT_DOUBLE_EQ(config.where_at_scoring.hit_radius_m, defaults.where_at_scoring.hit_radius_m);
    EXPECT_EQ(config.join.query_count, defaults.join.query_count);
    EXPECT_DOUBLE_EQ(config.join.eps_m, defaults.join.eps_m);
}

TEST(Config, ReadsJoinKeys) {
    const auto config = pipeline::load_config(write_config("join.json", R"({
        "data_path": "data.parquet",
        "join": {"query_count": 0, "eps_m": 250}
    })"));
    EXPECT_EQ(config.join.query_count, 0);
    EXPECT_DOUBLE_EQ(config.join.eps_m, 250);
}

TEST(Config, ReadsTopLevelAndNestedKeys) {
    const auto config = pipeline::load_config(write_config("overrides.json", R"({
        "data_path": "data.parquet",
        "seed": 123,
        "split": {"type": "Time", "query_percentage": 0.3},
        "budget_shares": [0.5],
        "algorithms": ["SED"],
        "knn": {"k": 3, "query_count": 0, "p": 0.25},
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
    EXPECT_DOUBLE_EQ(config.knn.p, 0.25);
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
    for (const auto &section : {"split", "knn", "range", "where_at", "join"}) {
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

TEST(Config, SelfJoinModeIsExplicitAndDefaultsToHeldOut) {
    EXPECT_EQ(pipeline::Config{}.join.mode, query::JoinMode::HeldOut);
    const auto config = pipeline::load_config(write_config("self_join.json", R"({
        "data_path": "data.parquet", "join": {"mode": "Self", "eps_m": 500}
    })"));
    EXPECT_EQ(config.join.mode, query::JoinMode::Self);
    EXPECT_FALSE(config.self_join);
    EXPECT_TRUE(pipeline::load_config(write_config("self_join_diagnostic.json", R"({
        "data_path": "data.parquet", "join": {"mode": "HeldOut", "self_join": true}
    })"))
                    .self_join);
    EXPECT_THROW(pipeline::load_config(write_config("bad_join_mode.json", R"({
        "data_path": "data.parquet", "join": {"mode": "typo"}
    })")),
                 std::invalid_argument);
}

TEST(Config, ReadsAllocationAlphas) {
    const auto config = pipeline::load_config(write_config(
        "alphas.json", R"({"data_path": "d", "allocation_alphas": [1.0, 0.25, 0.0]})"));
    EXPECT_EQ(config.allocation_alphas, (std::vector<double>{1.0, 0.25, 0.0}));
}

TEST(Config, RejectsEmptyAllocationAlphas) {
    EXPECT_THROW(pipeline::load_config(write_config(
                     "no_alphas.json", R"({"data_path": "d", "allocation_alphas": []})")),
                 std::invalid_argument);
}

TEST(Config, ReadsClusteringKeys) {
    const auto config = pipeline::load_config(write_config("clustering.json", R"({
        "data_path": "d", "clustering": {"enabled": true, "eps_m": 250.0, "min_size": 4}
    })"));
    EXPECT_TRUE(config.clustering.enabled);
    EXPECT_DOUBLE_EQ(config.clustering.eps_m, 250.0);
    EXPECT_EQ(config.clustering.min_size, 4u);
}

TEST(Config, RejectsInvalidClustering) {
    EXPECT_THROW(
        pipeline::load_config(write_config("clustering_small.json",
                                           R"({"data_path": "d", "clustering": {"min_size": 1}})")),
        std::invalid_argument);
    EXPECT_THROW(pipeline::load_config(write_config(
                     "clustering_key.json", R"({"data_path": "d", "clustering": {"minsize": 5}})")),
                 std::invalid_argument);
    for (const char *value : {"-1", "2.5", "\"5\""})
        EXPECT_THROW(
            pipeline::load_config(write_config(
                "clustering_bad.json",
                std::string(R"({"data_path": "d", "clustering": {"min_size": )") + value + "}}")),
            std::invalid_argument);
}

TEST(Config, FrechetVariantAppliesToEveryFrechetQuery) {
    const auto config = pipeline::load_config(
        write_config("frechet.json", R"({"data_path": "d", "frechet": "Discrete"})"));
    EXPECT_EQ(config.knn.frechet, query::FrechetVariant::Discrete);
    EXPECT_EQ(config.join.frechet, query::FrechetVariant::Discrete);
    EXPECT_EQ(config.clustering.frechet, query::FrechetVariant::Discrete);
    const auto defaults =
        pipeline::load_config(write_config("frechet_default.json", R"({"data_path": "d"})"));
    EXPECT_EQ(defaults.knn.frechet, query::FrechetVariant::Continuous);
    EXPECT_EQ(defaults.join.frechet, query::FrechetVariant::Continuous);
    EXPECT_EQ(defaults.clustering.frechet, query::FrechetVariant::Continuous);
    EXPECT_THROW(pipeline::load_config(
                     write_config("frechet_bad.json", R"({"data_path": "d", "frechet": "Weak"})")),
                 std::invalid_argument);
}

TEST(Config, RejectsInvalidWhereAtScoring) {
    for (const char *scoring : {R"({"min_tolerance_m": 500.0, "max_tolerance_m": 25.0})",
                                R"({"hit_radius_m": 0.0})", R"({"min_tolerance_m": -1.0})"})
        EXPECT_THROW(
            pipeline::load_config(write_config(
                "where_at_scoring_bad.json",
                std::string(R"({"data_path": "d", "where_at_scoring": )") + scoring + "}")),
            std::invalid_argument);
}

TEST(Config, TeacherIsOffByDefault) {
    const auto config =
        pipeline::load_config(write_config("teacher_default.json", R"({"data_path": "d"})"));
    EXPECT_FALSE(config.teacher.enabled);
    EXPECT_EQ(config.teacher.cache_dir, "cache");
    EXPECT_EQ(config.teacher.arms, (std::vector<std::string>{"TEACHER", "BR_TEACHER"}));
    EXPECT_TRUE(config.teacher.all_join_pairs);
    // The teacher's join term follows the join it is scored on
    EXPECT_FALSE(config.teacher.join);
    EXPECT_EQ(pipeline::teacher_objective_config(config).join_mode, query::JoinMode::HeldOut);
    const auto self = pipeline::load_config(
        write_config("teacher_self.json", R"({"data_path": "d", "join": {"mode": "Self"}})"));
    EXPECT_EQ(pipeline::teacher_objective_config(self).join_mode, query::JoinMode::Self);
}

TEST(Config, ReadsTheTeacherSection) {
    const auto config = pipeline::load_config(write_config("teacher.json", R"({
        "data_path": "d",
        "teacher": {"enabled": true, "cache_dir": "my_cache", "arms": ["GATEORIG_a0", "BR_a0"],
                    "range_query_count": 1000, "all_join_pairs": false, "shape_weight": 0.25,
                    "join": "Self"}
    })"));
    EXPECT_EQ(pipeline::teacher_objective_config(config).join_mode, query::JoinMode::Self);
    EXPECT_TRUE(config.teacher.enabled);
    EXPECT_EQ(config.teacher.cache_dir, "my_cache");
    EXPECT_EQ(config.teacher.arms, (std::vector<std::string>{"GATEORIG_a0", "BR_a0"}));
    EXPECT_EQ(config.teacher.range_query_count, 1000u);
    EXPECT_FALSE(config.teacher.all_join_pairs);
    EXPECT_DOUBLE_EQ(config.teacher.shape_weight, 0.25);
}

TEST(Config, RejectsBadTeacherSections) {
    for (const auto &section :
         {R"({"enabled": true, "arms": ["DP"]})", R"({"enabled": true, "arms": []})",
          R"({"arms": ["GATE_a0", "GATE_a0"]})", R"({"cache_dir": ""})",
          R"({"range_query_count": 0})", R"({"shape_weight": -1})", R"({"seed": 7})",
          R"({"join": "Both"})"}) {
        SCOPED_TRACE(section);
        const auto contents = std::string(R"({"data_path": "d", "teacher": )") + section + "}";
        EXPECT_THROW(pipeline::load_config(write_config("bad_teacher.json", contents)),
                     std::invalid_argument);
    }
}

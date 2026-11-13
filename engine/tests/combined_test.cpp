#include "backend/engine/evaluator/combined.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace {

constexpr double missing = std::numeric_limits<double>::quiet_NaN();

evaluator::Scoretable all_scores(double value) {
    evaluator::Scoretable scores{};
    scores.knn.recall_at_k = value;
    scores.knn.kendalls_tau_topk = value;
    scores.range_recall = value;
    scores.range_f1 = value;
    scores.where_at.within_radius = value;
    scores.where_at.average_hit = value;
    scores.join.f1 = value;
    return scores;
}

} // namespace

TEST(Normalize, MapsFloorToZeroAndPerfectToOne) {
    EXPECT_DOUBLE_EQ(evaluator::normalize(0.2, 0.2), 0.0);
    EXPECT_DOUBLE_EQ(evaluator::normalize(1.0, 0.2), 1.0);
    EXPECT_DOUBLE_EQ(evaluator::normalize(0.6, 0.2), 0.5);
}

TEST(Normalize, StretchesMetricsWithAHighFloor) {
    EXPECT_DOUBLE_EQ(evaluator::normalize(0.3, 0.2), 0.125);
    EXPECT_DOUBLE_EQ(evaluator::normalize(0.9, 0.8), 0.5);
}

TEST(Normalize, IsNegativeBelowFloor) { EXPECT_DOUBLE_EQ(evaluator::normalize(0.1, 0.2), -0.125); }

TEST(Normalize, IsNanWithoutRoomOrInput) {
    EXPECT_TRUE(std::isnan(evaluator::normalize(1.0, 1.0)));
    EXPECT_TRUE(std::isnan(evaluator::normalize(missing, 0.5)));
    EXPECT_TRUE(std::isnan(evaluator::normalize(0.5, missing)));
}

TEST(NormalizeScores, FloorScoresZeroAndPerfectScoresOne) {
    const auto floor = all_scores(0.4);
    EXPECT_DOUBLE_EQ(evaluator::normalize_scores(floor, floor).combined, 0.0);
    EXPECT_DOUBLE_EQ(evaluator::normalize_scores(all_scores(1.0), floor).combined, 1.0);
}

TEST(NormalizeScores, WeighsQueryTypesEquallyNotScores) {
    const auto floor = all_scores(0.0);
    auto scores = all_scores(0.0);
    scores.knn.recall_at_k = 1.0;
    scores.range_f1 = 1.0;
    const auto normalized = evaluator::normalize_scores(scores, floor);
    EXPECT_DOUBLE_EQ(normalized.knn_recall_at_k, 1.0);
    EXPECT_DOUBLE_EQ(normalized.knn, 1.0);
    EXPECT_DOUBLE_EQ(normalized.range, 1.0);
    EXPECT_DOUBLE_EQ(normalized.where_at, 0.0);
    EXPECT_DOUBLE_EQ(normalized.join, 0.0);
    // (1 + 1 + 0 + 0) / 4; diagnostics do not contribute.
    EXPECT_DOUBLE_EQ(normalized.combined, 0.5);
}

TEST(NormalizeScores, SkipsMissingQueryTypes) {
    auto floor = all_scores(0.5);
    auto scores = all_scores(0.75);
    floor.join.f1 = missing;
    scores.join.f1 = missing;
    scores.where_at.average_hit = missing;
    const auto normalized = evaluator::normalize_scores(scores, floor);
    EXPECT_TRUE(std::isnan(normalized.join));
    EXPECT_TRUE(std::isnan(normalized.where_at));
    EXPECT_DOUBLE_EQ(normalized.combined, 0.5);
}

TEST(NormalizeScores, AllMissingIsNan) {
    const auto normalized = evaluator::normalize_scores(all_scores(missing), all_scores(missing));
    EXPECT_TRUE(std::isnan(normalized.knn));
    EXPECT_TRUE(std::isnan(normalized.combined));
}

TEST(NormalizeScores, DiagnosticsNeverChangePrimaryScores) {
    const auto floor = all_scores(0.0);
    auto scores = all_scores(0.5);
    scores.knn.kendalls_tau_topk = -1.0;
    scores.range_recall = 0.0;
    scores.where_at.within_radius = 1.0;
    scores.join.micro_f1 = 0.0;
    const auto normalized = evaluator::normalize_scores(scores, floor);
    EXPECT_DOUBLE_EQ(normalized.combined, 0.5);
    EXPECT_DOUBLE_EQ(normalized.knn_kendalls_tau_topk, -1.0);
    EXPECT_DOUBLE_EQ(normalized.range_recall, 0.0);
    EXPECT_DOUBLE_EQ(normalized.where_at_within_radius, 1.0);
    scores.knn.recall_at_k = missing;
    scores.range_f1 = missing;
    scores.where_at.average_hit = missing;
    scores.join.f1 = missing;
    EXPECT_TRUE(std::isnan(evaluator::normalize_scores(scores, floor).combined));
}

TEST(NormalizeScores, SelfJoinIsNormalisedButStaysOutOfCombined) {
    auto floor = all_scores(0.0);
    auto scores = all_scores(0.5);
    floor.self_join.f1 = 0.2;
    scores.self_join.f1 = 0.6;
    const auto normalized = evaluator::normalize_scores(scores, floor);
    EXPECT_DOUBLE_EQ(normalized.self_join, 0.5);
    EXPECT_DOUBLE_EQ(normalized.join, 0.5);
    scores.self_join.f1 = 1.0;
    EXPECT_DOUBLE_EQ(evaluator::normalize_scores(scores, floor).combined, normalized.combined);
    EXPECT_TRUE(std::isnan(evaluator::normalize_scores(all_scores(0.5), floor).self_join));
}

TEST(NormalizeScores, WhereAtUsesAverageHitAndItsOwnEndpointFloor) {
    auto floor = all_scores(0.0);
    auto scores = all_scores(0.0);
    floor.where_at.within_radius = 0.2;
    scores.where_at.within_radius = 0.6;
    floor.where_at.average_hit = 0.4;
    scores.where_at.average_hit = 1.0;
    const auto normalized = evaluator::normalize_scores(scores, floor);
    EXPECT_DOUBLE_EQ(normalized.where_at, 1.0);
    EXPECT_DOUBLE_EQ(normalized.where_at_within_radius, 0.5);
    EXPECT_DOUBLE_EQ(normalized.combined, 0.25);
}

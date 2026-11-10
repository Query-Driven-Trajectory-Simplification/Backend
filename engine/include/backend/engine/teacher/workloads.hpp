#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/where_at_query.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace teacher {

// The teacher's training workloads. They come from the database alone, never from the held-out
// query pool, and use their own seeds, so the teacher is never scored on what it trained on
// (except the self-join, which is database-wide by nature; hence the seen/unseen pair split).
struct WorkloadConfig {
    std::size_t k = 10;          // kNN neighbours per database query
    double join_eps_m = 150.0;   // self-join threshold
    query::FrechetVariant frechet = query::FrechetVariant::Continuous;
    query::RangeConfig range{.query_count = 64000};
    query::WhereAtConfig where_at;
};

// A database trajectory used as a kNN query against the rest, itself excluded
struct KnnEntry {
    bool valid = false;              // false with fewer than k + 1 neighbours
    std::vector<std::uint32_t> top;  // the k nearest, closest first
    double cutoff = 0.0;             // Fréchet distance to the (k + 1)th neighbour
};

struct JoinPair {
    std::uint32_t i, j; // i < j
    bool seen;          // only seen pairs are given to the teacher
};

struct TeacherWorkloads {
    std::size_t k = 0;
    double join_eps_m = 0.0;
    query::FrechetVariant frechet = query::FrechetVariant::Continuous;
    std::vector<KnnEntry> knn;       // one per database trajectory
    std::vector<JoinPair> positives; // self-join pairs, half seen
    std::vector<JoinPair> negatives; // endpoint candidates that are not pairs, half seen
    query::RangeWorkload range;
    query::WhereAtWorkload where_at;
};

// Loads the kNN and join part from cache_dir/teacher_cache_seed<seed>.bin if it matches the
// database, seed and settings, otherwise builds it (the slow part, minutes to an hour) and saves
// it. Range and where-at are regenerated every time
TeacherWorkloads load_or_build_workloads(const core::TrajectoryDb &db,
                                         const WorkloadConfig &config, std::uint64_t seed,
                                         const std::filesystem::path &cache_dir);

} // namespace teacher

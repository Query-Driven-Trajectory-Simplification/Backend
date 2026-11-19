#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/where_at_query.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace teacher {

// Offsets added to the run seed, one per random draw, so no two draws share a stream
inline constexpr std::uint64_t kRangeSeedOffset = 7001;
inline constexpr std::uint64_t kWhereAtSeedOffset = 7002;
inline constexpr std::uint64_t kPairSplitSeedOffset = 7003;
inline constexpr std::uint64_t kOrderSeedOffset = 7004;
inline constexpr std::uint64_t kNegativeSeedOffset = 7005;
inline constexpr std::uint64_t kShuffleSeedOffset = 7006;

inline constexpr std::size_t kRangeQueryCount = 64000;

// Training workloads from the database alone, with their own seeds, never the held-out pool
struct WorkloadConfig {
    std::size_t k = 10;
    double join_eps_m = 150.0;
    std::vector<double> extra_join_eps_m;
    query::FrechetVariant frechet = query::FrechetVariant::Continuous;
    query::RangeConfig range{.query_count = kRangeQueryCount};
    query::WhereAtConfig where_at;
};

// A database trajectory used as a kNN query against the rest, itself excluded
struct KnnEntry {
    bool valid = false;             // false with fewer than k + 1 neighbours
    std::vector<std::uint32_t> top; // the k nearest, closest first
    double cutoff = 0.0;            // Fréchet distance to the (k + 1)th neighbour
};

struct JoinPair {
    std::uint32_t i, j; // i < j
    bool seen;          // in the half used with all_join_pairs = false
};

// Self-join pairs at a threshold other than join_eps_m, all of them used
struct ExtraJoin {
    double eps_m = 0.0;
    std::vector<JoinPair> positives, negatives;
};

struct TeacherWorkloads {
    std::size_t k = 0;
    double join_eps_m = 0.0;
    query::FrechetVariant frechet = query::FrechetVariant::Continuous;
    std::vector<KnnEntry> knn;       // one per database trajectory
    std::vector<JoinPair> positives; // self-join pairs, half seen
    std::vector<JoinPair> negatives; // endpoint candidates that are not pairs, half seen
    std::vector<ExtraJoin> extra_joins;
    query::RangeWorkload range;
    query::WhereAtWorkload where_at;
};

// kNN and join pairs come from cache_dir/teacher_cache_seed<seed>.bin when it matches the database
// and settings, otherwise they are built (slow) and saved. The rest is regenerated every time
TeacherWorkloads load_or_build_workloads(const core::TrajectoryDb &db, const WorkloadConfig &config,
                                         std::uint64_t seed,
                                         const std::filesystem::path &cache_dir);

// The self-kNN test workload from the cached lists: every valid database query with its k nearest
// others. They depend on the database alone, so they are what a fresh build would give
query::KnnWorkload self_knn_workload(const TeacherWorkloads &workloads, double p);

} // namespace teacher

#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/where_at_query.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace teacher {

// Kept point indices per trajectory, sorted, endpoints included
using KeptSets = std::vector<std::vector<std::size_t>>;

inline constexpr double kJoinEps = 150.0;
inline constexpr std::uint64_t kRangeSeedOffset = 7001;
inline constexpr std::uint64_t kWhereAtSeedOffset = 7002;
inline constexpr std::uint64_t kPairSplitSeedOffset = 7003;

// Same as the engine's total_budget
std::size_t total_budget(const core::TrajectoryDb &db, double share);

struct KnnTeacherQuery {
    bool valid = false;               // false with fewer than 11 neighbours
    std::vector<std::uint32_t> top10; // closest first
    double d11 = 0.0;                 // continuous Fréchet to the 11th neighbour
};

struct JoinPair {
    std::uint32_t i, j; // i < j
    bool seen;
};

// Built from the database alone
struct TeacherCaches {
    std::uint64_t seed = 0;
    std::vector<KnnTeacherQuery> knn; // each database trajectory as a query
    std::vector<JoinPair> pairs;      // 150 m self-join, half of them seen
    query::RangeWorkload range;       // 64k boxes
    query::WhereAtWorkload where_at;  // 128 probes per trajectory
};

// Reads the kNN and pair cache from dir, or builds and saves it. Range and where-at are
// regenerated each time
TeacherCaches load_or_build_caches(const core::TrajectoryDb &db, std::uint64_t seed,
                                   const std::filesystem::path &dir);

struct TeacherOutput {
    std::map<std::string, KeptSets> arms;
    // J_T for any kept sets, so it can be compared on the baselines too
    std::function<double(const KeptSets &, double share)> objective;
    std::map<std::string, double> timings_s; // written to timings.csv
};

// Gets no held-out data. Arm names end in "_" + share_tag(share)
TeacherOutput run_teacher(const core::TrajectoryDb &db, const TeacherCaches &caches,
                          std::uint64_t seed, std::span<const double> shares);

std::string share_tag(double share);

// RDP_a0, RDP_a025, SED_a0, UNI_a0, PARTNER_l0.5 and PARTNER_l1
std::map<std::string, KeptSets> baseline_arms(const core::TrajectoryDb &db,
                                              const TeacherCaches &caches,
                                              std::span<const double> shares);

// Shares the interior points by weight, capped per trajectory. Budgets include endpoints
std::vector<std::size_t> weighted_budgets(const core::TrajectoryDb &db,
                                          const std::vector<double> &w, std::size_t interior_total);

} // namespace teacher

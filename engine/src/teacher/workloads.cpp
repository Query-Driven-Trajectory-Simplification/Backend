#include "backend/engine/teacher/workloads.hpp"

#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/timer.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace teacher {

namespace {

// Bump when the layout or workload generation changes. V5 drops the seen / unseen split
constexpr std::uint64_t kMagicV5 = 0x5431'4341'4348'4535ull;
constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

// FNV-1a over IDs, sizes and every point, projected coordinates included
std::uint64_t fingerprint(const core::TrajectoryDb &db) {
    std::uint64_t hash = kFnvOffsetBasis;
    auto mix = [&](std::uint64_t value) {
        for (unsigned int byte = 0; byte < 8; ++byte) {
            hash ^= value & 0xff;
            hash *= kFnvPrime;
            value >>= 8;
        }
    };
    mix(db.size());
    mix(db.num_points());
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto trajectory = db.trajectory(i);
        mix(static_cast<std::uint64_t>(trajectory.trajectory_id));
        mix(trajectory.size());
        for (std::size_t k = 0; k < trajectory.size(); ++k) {
            mix(static_cast<std::uint64_t>(trajectory.t[k]));
            mix(std::bit_cast<std::uint64_t>(trajectory.lat[k]));
            mix(std::bit_cast<std::uint64_t>(trajectory.lon[k]));
            mix(std::bit_cast<std::uint64_t>(trajectory.x[k]));
            mix(std::bit_cast<std::uint64_t>(trajectory.y[k]));
        }
    }
    return hash;
}

template <typename T> void put(std::ofstream &out, const T &value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}
template <typename T> T get(std::ifstream &in) {
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    return value;
}

std::vector<KnnEntry> build_knn(const core::TrajectoryDb &db, const WorkloadConfig &config) {
    std::vector<KnnEntry> result(db.size());
    core::parallel_for(db.size(), [&](std::size_t e) {
        const auto query = db.trajectory(e);
        auto neighbours = query::knn(db, query, config.k + 2, config.frechet);
        std::erase(neighbours, e);
        auto &entry = result[e];
        if (neighbours.size() < config.k + 1)
            return;
        entry.cutoff =
            query::frechet_distance(query, db.trajectory(neighbours[config.k]), config.frechet);
        entry.top.assign(neighbours.begin(),
                         neighbours.begin() + static_cast<std::ptrdiff_t>(config.k));
        entry.valid = std::isfinite(entry.cutoff);
        if (!entry.valid)
            entry.top.clear();
    });
    return result;
}

// Positives and negatives from one self-join
void collect_pairs(const core::TrajectoryDb &db, double eps_m, query::FrechetVariant frechet,
                   std::vector<JoinPair> &positives, std::vector<JoinPair> &negatives) {
    const auto join = query::generate_self_join(db, eps_m, frechet);
    positives.clear();
    negatives.clear();
    for (const auto &entry : join.entries)
        for (const auto j : entry.candidates) {
            const JoinPair pair{static_cast<std::uint32_t>(entry.query_index),
                                static_cast<std::uint32_t>(j)};
            if (std::ranges::binary_search(entry.reference, j))
                positives.push_back(pair);
            else
                negatives.push_back(pair);
        }
}

void write_pairs(std::ofstream &out, const std::vector<JoinPair> &pairs) {
    put<std::uint64_t>(out, pairs.size());
    for (const auto &pair : pairs) {
        put(out, pair.i);
        put(out, pair.j);
    }
}

void read_pairs(std::ifstream &in, std::vector<JoinPair> &pairs) {
    pairs.resize(get<std::uint64_t>(in));
    for (auto &pair : pairs) {
        pair.i = get<std::uint32_t>(in);
        pair.j = get<std::uint32_t>(in);
    }
}

void save(const std::filesystem::path &file, std::uint64_t seed, std::uint64_t db_fingerprint,
          const TeacherWorkloads &workloads) {
    if (file.has_parent_path())
        std::filesystem::create_directories(file.parent_path());
    const auto tmp = file.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        put(out, kMagicV5);
        put(out, seed);
        put(out, db_fingerprint);
        put<std::uint64_t>(out, workloads.k);
        put(out, workloads.join_eps_m);
        put<std::uint8_t>(out, static_cast<std::uint8_t>(workloads.frechet));
        put<std::uint64_t>(out, workloads.knn.size());
        for (const auto &entry : workloads.knn) {
            put<std::uint8_t>(out, entry.valid);
            put(out, entry.cutoff);
            for (std::size_t k = 0; k < workloads.k; ++k)
                put<std::uint32_t>(out, k < entry.top.size() ? entry.top[k] : 0);
        }
        write_pairs(out, workloads.positives);
        write_pairs(out, workloads.negatives);
        if (!out)
            throw std::runtime_error("teacher cache: write failed " + tmp);
    }
    std::filesystem::rename(tmp, file);
}

bool load(const std::filesystem::path &file, std::uint64_t seed, std::uint64_t db_fingerprint,
          const WorkloadConfig &config, TeacherWorkloads &workloads) {
    std::ifstream in(file, std::ios::binary);
    if (!in)
        return false;
    if (get<std::uint64_t>(in) != kMagicV5 || get<std::uint64_t>(in) != seed ||
        get<std::uint64_t>(in) != db_fingerprint)
        return false;
    const auto k = get<std::uint64_t>(in);
    const auto eps = get<double>(in);
    const auto variant = static_cast<query::FrechetVariant>(get<std::uint8_t>(in));
    if (k != config.k || eps != config.join_eps_m || variant != config.frechet)
        return false;
    workloads.knn.resize(get<std::uint64_t>(in));
    for (auto &entry : workloads.knn) {
        entry.valid = get<std::uint8_t>(in) != 0;
        entry.cutoff = get<double>(in);
        entry.top.resize(k);
        for (auto &neighbour : entry.top)
            neighbour = get<std::uint32_t>(in);
        if (!entry.valid)
            entry.top.clear();
    }
    read_pairs(in, workloads.positives);
    read_pairs(in, workloads.negatives);
    return static_cast<bool>(in);
}

} // namespace

TeacherWorkloads load_or_build_workloads(const core::TrajectoryDb &db, const WorkloadConfig &config,
                                         std::uint64_t seed,
                                         const std::filesystem::path &cache_dir) {
    if (config.k == 0)
        throw std::invalid_argument("teacher workloads: k must be positive");
    query::check_extra_thresholds(config.extra_join_eps_m, config.join_eps_m,
                                  "teacher workloads: extra_join_eps_m");
    TeacherWorkloads workloads;
    workloads.k = config.k;
    workloads.join_eps_m = config.join_eps_m;
    workloads.frechet = config.frechet;
    const auto db_fingerprint = fingerprint(db);
    const auto file = cache_dir / std::format("teacher_cache_seed{}.bin", seed);

    if (load(file, seed, db_fingerprint, config, workloads) && workloads.knn.size() == db.size()) {
        std::cout << "teacher: cache loaded from " << file.string() << '\n';
    } else {
        std::cout << "teacher: no matching cache at " << file.string()
                  << ", building it (database-as-query kNN takes a while)\n";
        auto start = std::chrono::steady_clock::now();
        workloads.knn = build_knn(db, config);
        std::cout << std::format("teacher: kNN {:.1f} s\n", core::seconds_since(start));
        start = std::chrono::steady_clock::now();
        collect_pairs(db, config.join_eps_m, config.frechet, workloads.positives,
                      workloads.negatives);
        std::cout << std::format("teacher: self-join pairs {:.1f} s\n", core::seconds_since(start));
        save(file, seed, db_fingerprint, workloads);
        std::cout << "teacher: cache saved to " << file.string() << '\n';
    }

    workloads.range = query::generate_range_queries(db, config.range, seed + kRangeSeedOffset);
    workloads.where_at =
        query::generate_where_at_queries(db, config.where_at, seed + kWhereAtSeedOffset);
    for (const double eps : config.extra_join_eps_m) {
        ExtraJoin extra{.eps_m = eps, .positives = {}, .negatives = {}};
        collect_pairs(db, eps, config.frechet, extra.positives, extra.negatives);
        std::cout << std::format("teacher: extra join at {} m, {} pairs, {} negatives\n", eps,
                                 extra.positives.size(), extra.negatives.size());
        workloads.extra_joins.push_back(std::move(extra));
    }

    std::size_t invalid = 0;
    for (const auto &entry : workloads.knn)
        invalid += !entry.valid;
    std::cout << std::format("teacher: {} kNN queries ({} invalid), {} pairs, {} negatives, {} "
                             "boxes, {} probes\n",
                             workloads.knn.size(), invalid, workloads.positives.size(),
                             workloads.negatives.size(), workloads.range.queries.size(),
                             workloads.where_at.queries.size());
    return workloads;
}

query::KnnWorkload self_knn_workload(const TeacherWorkloads &workloads, double p) {
    query::KnnWorkload result{.queries = core::TrajectoryDb({0}, {}, {}, {}, {}),
                              .entries = {},
                              .k = workloads.k,
                              .p = p,
                              .frechet = workloads.frechet,
                              .self = true};
    for (std::size_t e = 0; e < workloads.knn.size(); ++e)
        if (workloads.knn[e].valid)
            result.entries.push_back({e, std::vector<std::size_t>(workloads.knn[e].top.begin(),
                                                                  workloads.knn[e].top.end())});
    return result;
}

} // namespace teacher

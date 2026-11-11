#include "backend/engine/teacher/workloads.hpp"

#include "backend/engine/core/parallel.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace teacher {

namespace {

// Versions cover both the layout and cached workload generation; bump when either changes.
// Version 4 fingerprints every point. Older caches have only a metadata fingerprint and must
// be rebuilt, since their kNN entries and pair truth cannot be verified against the geometry.
constexpr std::uint64_t kMagicV4 = 0x5431'4341'4348'4534ull;
constexpr std::uint64_t kRangeSeedOffset = 7001;
constexpr std::uint64_t kWhereAtSeedOffset = 7002;
constexpr std::uint64_t kPairSplitSeedOffset = 7003;
constexpr std::uint64_t kNegativeSeedOffset = 7005;

// FNV-1a over trajectory boundaries, IDs and every point's time and coordinates. Include the
// projected coordinates too, so changing the projection invalidates cached spatial queries.
std::uint64_t fingerprint(const core::TrajectoryDb &db) {
    std::uint64_t h = 14695981039346656037ull;
    auto mix = [&](std::uint64_t v) {
        for (unsigned int byte = 0; byte < 8; ++byte) {
            h ^= v & 0xff;
            h *= 1099511628211ull;
            v >>= 8;
        }
    };
    mix(db.size());
    mix(db.num_points());
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto v = db.trajectory(i);
        mix(static_cast<std::uint64_t>(v.trajectory_id));
        mix(v.size());
        for (std::size_t k = 0; k < v.size(); ++k) {
            mix(static_cast<std::uint64_t>(v.t[k]));
            mix(std::bit_cast<std::uint64_t>(v.lat[k]));
            mix(std::bit_cast<std::uint64_t>(v.lon[k]));
            mix(std::bit_cast<std::uint64_t>(v.x[k]));
            mix(std::bit_cast<std::uint64_t>(v.y[k]));
        }
    }
    return h;
}

template <typename T> void put(std::ofstream &out, const T &v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof(T));
}
template <typename T> T get(std::ifstream &in) {
    T v{};
    in.read(reinterpret_cast<char *>(&v), sizeof(T));
    return v;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

double distance(const core::TrajectoryView &a, const core::TrajectoryView &b,
                query::FrechetVariant variant) {
    const double discrete = query::frechet_distance(a, b);
    // Discrete Fréchet bounds the continuous distance from above
    return variant == query::FrechetVariant::Continuous
               ? query::continuous_frechet_distance(a, b, 0.0, discrete)
               : discrete;
}

std::vector<KnnEntry> build_knn(const core::TrajectoryDb &db, const WorkloadConfig &config) {
    std::vector<KnnEntry> result(db.size());
    core::parallel_for(db.size(), [&](std::size_t e) {
        const auto query = db.trajectory(e);
        auto nb = query::knn(db, query, config.k + 2, config.frechet);
        std::erase(nb, e);
        auto &r = result[e];
        if (nb.size() < config.k + 1)
            return;
        r.cutoff = distance(query, db.trajectory(nb[config.k]), config.frechet);
        r.top.assign(nb.begin(), nb.begin() + static_cast<std::ptrdiff_t>(config.k));
        r.valid = std::isfinite(r.cutoff);
        if (!r.valid)
            r.top.clear();
    });
    return result;
}

void split_seen(std::vector<JoinPair> &pairs, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::shuffle(pairs.begin(), pairs.end(), rng);
    for (std::size_t k = 0; k < pairs.size(); ++k)
        pairs[k].seen = k < pairs.size() / 2;
    std::ranges::sort(pairs, [](const JoinPair &a, const JoinPair &b) {
        return a.i < b.i || (a.i == b.i && a.j < b.j);
    });
}

// Positives and negatives from one self-join, each split into seen and unseen halves
void build_pairs(const core::TrajectoryDb &db, const WorkloadConfig &config, std::uint64_t seed,
                 std::vector<JoinPair> *positives, std::vector<JoinPair> &negatives) {
    const auto join = query::generate_self_join(db, config.join_eps_m, config.frechet);
    if (positives)
        positives->clear();
    negatives.clear();
    for (const auto &entry : join.entries)
        for (const auto j : entry.candidates) {
            const JoinPair pair{static_cast<std::uint32_t>(entry.query_index),
                                static_cast<std::uint32_t>(j), false};
            if (std::ranges::binary_search(entry.reference, j)) {
                if (positives)
                    positives->push_back(pair);
            } else {
                negatives.push_back(pair);
            }
        }
    if (positives)
        split_seen(*positives, seed + kPairSplitSeedOffset);
    split_seen(negatives, seed + kNegativeSeedOffset);
}

void write_pairs(std::ofstream &out, const std::vector<JoinPair> &pairs) {
    put<std::uint64_t>(out, pairs.size());
    for (const auto &p : pairs) {
        put(out, p.i);
        put(out, p.j);
        put<std::uint8_t>(out, p.seen);
    }
}

void read_pairs(std::ifstream &in, std::vector<JoinPair> &pairs) {
    pairs.resize(get<std::uint64_t>(in));
    for (auto &p : pairs) {
        p.i = get<std::uint32_t>(in);
        p.j = get<std::uint32_t>(in);
        p.seen = get<std::uint8_t>(in) != 0;
    }
}

void save(const std::filesystem::path &file, std::uint64_t seed, std::uint64_t print,
          const TeacherWorkloads &w) {
    if (file.has_parent_path())
        std::filesystem::create_directories(file.parent_path());
    const auto tmp = file.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        put(out, kMagicV4);
        put(out, seed);
        put(out, print);
        put<std::uint64_t>(out, w.k);
        put(out, w.join_eps_m);
        put<std::uint8_t>(out, static_cast<std::uint8_t>(w.frechet));
        put<std::uint64_t>(out, w.knn.size());
        for (const auto &q : w.knn) {
            put<std::uint8_t>(out, q.valid);
            put(out, q.cutoff);
            for (std::size_t k = 0; k < w.k; ++k)
                put<std::uint32_t>(out, k < q.top.size() ? q.top[k] : 0);
        }
        write_pairs(out, w.positives);
        write_pairs(out, w.negatives);
        if (!out)
            throw std::runtime_error("teacher cache: write failed " + tmp);
    }
    std::filesystem::rename(tmp, file);
}

// Reads only the current workload version, matching all point contents, seed and settings.
bool load(const std::filesystem::path &file, std::uint64_t seed, std::uint64_t print,
          const WorkloadConfig &config, TeacherWorkloads &w) {
    std::ifstream in(file, std::ios::binary);
    if (!in)
        return false;
    if (get<std::uint64_t>(in) != kMagicV4 || get<std::uint64_t>(in) != seed ||
        get<std::uint64_t>(in) != print)
        return false;
    const auto k = get<std::uint64_t>(in);
    const auto eps = get<double>(in);
    const auto variant = static_cast<query::FrechetVariant>(get<std::uint8_t>(in));
    if (k != config.k || eps != config.join_eps_m || variant != config.frechet)
        return false;
    w.knn.resize(get<std::uint64_t>(in));
    for (auto &q : w.knn) {
        q.valid = get<std::uint8_t>(in) != 0;
        q.cutoff = get<double>(in);
        q.top.resize(k);
        for (auto &v : q.top)
            v = get<std::uint32_t>(in);
        if (!q.valid)
            q.top.clear();
    }
    read_pairs(in, w.positives);
    read_pairs(in, w.negatives);
    return static_cast<bool>(in);
}

} // namespace

TeacherWorkloads load_or_build_workloads(const core::TrajectoryDb &db, const WorkloadConfig &config,
                                         std::uint64_t seed,
                                         const std::filesystem::path &cache_dir) {
    if (config.k == 0)
        throw std::invalid_argument("teacher workloads: k must be positive");
    TeacherWorkloads w;
    w.k = config.k;
    w.join_eps_m = config.join_eps_m;
    w.frechet = config.frechet;
    const auto print = fingerprint(db);
    const auto file = cache_dir / std::format("teacher_cache_seed{}.bin", seed);

    if (load(file, seed, print, config, w) && w.knn.size() == db.size()) {
        std::cout << "teacher: cache loaded from " << file.string() << '\n';
    } else {
        std::cout << "teacher: no matching cache at " << file.string()
                  << ", building it (database-as-query kNN takes a while)\n";
        auto start = std::chrono::steady_clock::now();
        w.knn = build_knn(db, config);
        std::cout << std::format("teacher: kNN {:.1f} s\n", seconds_since(start));
        start = std::chrono::steady_clock::now();
        build_pairs(db, config, seed, &w.positives, w.negatives);
        std::cout << std::format("teacher: self-join pairs {:.1f} s\n", seconds_since(start));
        save(file, seed, print, w);
        std::cout << "teacher: cache saved to " << file.string() << '\n';
    }

    w.range = query::generate_range_queries(db, config.range, seed + kRangeSeedOffset);
    w.where_at = query::generate_where_at_queries(db, config.where_at, seed + kWhereAtSeedOffset);

    std::size_t invalid = 0;
    for (const auto &q : w.knn)
        invalid += !q.valid;
    std::cout << std::format("teacher: {} kNN queries ({} invalid), {} pairs, {} negatives, {} "
                             "boxes, {} probes\n",
                             w.knn.size(), invalid, w.positives.size(), w.negatives.size(),
                             w.range.queries.size(), w.where_at.queries.size());
    return w;
}

} // namespace teacher

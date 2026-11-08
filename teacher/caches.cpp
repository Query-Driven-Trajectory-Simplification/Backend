#include "teacher.hpp"

#include "backend/engine/core/parallel.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace teacher {

namespace {

constexpr std::uint64_t kMagic = 0x5431'4341'4348'4532ull; // format version 2

// Changes when the database does
std::uint64_t fingerprint(const core::TrajectoryDb &db) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](std::uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(db.size());
    mix(db.num_points());
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto v = db.trajectory(i);
        mix(static_cast<std::uint64_t>(v.trajectory_id));
        mix(v.size());
        mix(static_cast<std::uint64_t>(v.t.front()));
        mix(static_cast<std::uint64_t>(v.t.back()));
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

std::vector<KnnTeacherQuery> build_knn(const core::TrajectoryDb &db) {
    std::vector<KnnTeacherQuery> result(db.size());
    core::parallel_for(db.size(), [&](std::size_t e) {
        const auto query = db.trajectory(e);
        auto nb = query::knn(db, query, 12, query::FrechetVariant::Continuous);
        std::erase(nb, e);
        auto &r = result[e];
        if (nb.size() < 11)
            return;
        const auto eleventh = db.trajectory(nb[10]);
        r.d11 = query::continuous_frechet_distance(query, eleventh, 0.0,
                                                   query::frechet_distance(query, eleventh));
        r.top10.assign(nb.begin(), nb.begin() + 10);
        r.valid = std::isfinite(r.d11);
    });
    return result;
}

std::vector<JoinPair> build_pairs(const core::TrajectoryDb &db, std::uint64_t seed) {
    const auto join = query::generate_self_join(db, kJoinEps, query::FrechetVariant::Continuous);
    std::vector<JoinPair> pairs;
    for (const auto &entry : join.entries)
        for (const auto j : entry.reference)
            pairs.push_back({static_cast<std::uint32_t>(entry.query_index),
                             static_cast<std::uint32_t>(j), false});
    std::mt19937_64 rng(seed + kPairSplitSeedOffset);
    std::shuffle(pairs.begin(), pairs.end(), rng);
    for (std::size_t k = 0; k < pairs.size() / 2; ++k)
        pairs[k].seen = true;
    std::ranges::sort(pairs, [](const JoinPair &a, const JoinPair &b) {
        return a.i < b.i || (a.i == b.i && a.j < b.j);
    });
    return pairs;
}

void save(const std::filesystem::path &file, std::uint64_t seed, std::uint64_t print,
          const std::vector<KnnTeacherQuery> &knn, const std::vector<JoinPair> &pairs) {
    std::filesystem::create_directories(file.parent_path());
    const auto tmp = file.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        put(out, kMagic);
        put(out, seed);
        put(out, print);
        put<std::uint64_t>(out, knn.size());
        for (const auto &q : knn) {
            put<std::uint8_t>(out, q.valid);
            put(out, q.d11);
            for (std::size_t k = 0; k < 10; ++k)
                put<std::uint32_t>(out, k < q.top10.size() ? q.top10[k] : 0);
        }
        put<std::uint64_t>(out, pairs.size());
        for (const auto &p : pairs) {
            put(out, p.i);
            put(out, p.j);
            put<std::uint8_t>(out, p.seen);
        }
        if (!out)
            throw std::runtime_error("cache: write failed " + tmp);
    }
    std::filesystem::rename(tmp, file);
}

bool load(const std::filesystem::path &file, std::uint64_t seed, std::uint64_t print,
          std::vector<KnnTeacherQuery> &knn, std::vector<JoinPair> &pairs) {
    std::ifstream in(file, std::ios::binary);
    if (!in || get<std::uint64_t>(in) != kMagic || get<std::uint64_t>(in) != seed ||
        get<std::uint64_t>(in) != print)
        return false;
    knn.resize(get<std::uint64_t>(in));
    for (auto &q : knn) {
        q.valid = get<std::uint8_t>(in) != 0;
        q.d11 = get<double>(in);
        q.top10.resize(10);
        for (auto &v : q.top10)
            v = get<std::uint32_t>(in);
        if (!q.valid)
            q.top10.clear();
    }
    pairs.resize(get<std::uint64_t>(in));
    for (auto &p : pairs) {
        p.i = get<std::uint32_t>(in);
        p.j = get<std::uint32_t>(in);
        p.seen = get<std::uint8_t>(in) != 0;
    }
    return static_cast<bool>(in);
}

} // namespace

std::size_t total_budget(const core::TrajectoryDb &db, double share) {
    const std::size_t endpoints = 2 * db.size();
    const double extra = share * static_cast<double>(db.num_points() - endpoints);
    return endpoints + static_cast<std::size_t>(std::llround(extra));
}

std::string share_tag(double share) { return std::format("share{:g}", share); }

TeacherCaches load_or_build_caches(const core::TrajectoryDb &db, std::uint64_t seed,
                                   const std::filesystem::path &dir) {
    TeacherCaches caches;
    caches.seed = seed;
    const auto print = fingerprint(db);
    const auto file = dir / std::format("teacher_cache_seed{}.bin", seed);
    if (load(file, seed, print, caches.knn, caches.pairs) && caches.knn.size() == db.size()) {
        std::cerr << "cache: loaded " << file << '\n';
    } else {
        auto start = std::chrono::steady_clock::now();
        caches.knn = build_knn(db);
        std::cerr << std::format("cache: database-as-query kNN {:.1f} s\n", seconds_since(start));
        start = std::chrono::steady_clock::now();
        caches.pairs = build_pairs(db, seed);
        std::cerr << std::format("cache: self-join pairs {:.1f} s\n", seconds_since(start));
        save(file, seed, print, caches.knn, caches.pairs);
        std::cerr << "cache: saved " << file << '\n';
    }
    caches.range = query::generate_range_queries(db, {64000, 2000.0, 604800, 5, 20, 1000},
                                                 seed + kRangeSeedOffset);
    caches.where_at = query::generate_where_at_queries(db, {128, 60.0}, seed + kWhereAtSeedOffset);
    std::size_t invalid = 0, seen = 0;
    for (const auto &q : caches.knn)
        invalid += !q.valid;
    for (const auto &p : caches.pairs)
        seen += p.seen;
    std::cerr << std::format("cache: {} kNN queries ({} invalid), {} pairs ({} seen), {} boxes, "
                             "{} probes\n",
                             caches.knn.size(), invalid, caches.pairs.size(), seen,
                             caches.range.queries.size(), caches.where_at.queries.size());
    return caches;
}

} // namespace teacher

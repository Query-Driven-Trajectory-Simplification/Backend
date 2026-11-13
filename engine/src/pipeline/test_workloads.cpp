#include "backend/engine/pipeline/test_workloads.hpp"

#include <magic_enum/magic_enum.hpp>

#include <cstddef>
#include <iostream>

namespace pipeline {

namespace {

query::KnnWorkload make_knn(const core::TrajectoryDb &database, const core::TrajectoryDb &queries,
                            const Config &config) {
    auto knn = query::generate_knn_queries(database, queries, config.knn, config.seed);
    std::cout << "knn: " << knn.entries.size() << " queries, k=" << knn.k << '\n';
    return knn;
}

query::WhereAtWorkload make_where_at(const core::TrajectoryDb &database, const Config &config) {
    auto where_at = query::generate_where_at_queries(database, config.where_at, config.seed);
    std::cout << "where-at: " << where_at.queries.size() << " probes, "
              << where_at.skipped_trajectories << " trips without eligible time\n";
    return where_at;
}

query::RangeWorkload make_range(const core::TrajectoryDb &database, const Config &config) {
    auto range = query::generate_range_queries(database, config.range, config.seed);
    std::cout << "range: " << range.queries.size() << " queries\n";
    return range;
}

void print_join(const query::JoinWorkload &join) {
    std::size_t pairs = 0;
    for (const auto &entry : join.entries)
        pairs += entry.reference.size();
    std::cout << "join " << magic_enum::enum_name(join.mode) << ": " << join.entries.size()
              << (join.mode == query::JoinMode::Self ? " database trajectories, eps="
                                                     : " queries, eps=")
              << join.eps_m << " m, " << pairs << " matching pairs\n";
}

query::JoinWorkload make_join(const core::TrajectoryDb &database, const core::TrajectoryDb &queries,
                              const Config &config) {
    auto join = query::generate_join_queries(database, queries, config.join, config.seed);
    print_join(join);
    return join;
}

// Empty when the join already is the self join
std::optional<query::JoinWorkload> make_self_join(const core::TrajectoryDb &database,
                                                  const Config &config) {
    if (!config.self_join || config.join.mode == query::JoinMode::Self)
        return std::nullopt;
    auto join = query::generate_self_join(database, config.join.eps_m, config.join.frechet);
    print_join(join);
    return join;
}

std::optional<query::RouteClusteringWorkload> make_clustering(const core::TrajectoryDb &database,
                                                              const query::JoinWorkload &join,
                                                              const Config &config) {
    if (!config.clustering.enabled)
        return std::nullopt;
    auto clustering = query::generate_route_clustering(database, config.clustering, &join);
    std::size_t covered = 0;
    for (const auto &members : clustering.reference.members)
        covered += members.size();
    std::cout << "route clustering: " << clustering.reference.members.size()
              << " clusters covering " << covered << " trajectories, eps=" << clustering.eps_m
              << " m, min size " << clustering.min_size << '\n';
    return clustering;
}

} // namespace

TestWorkloads::TestWorkloads(const core::TrajectoryDb &database, const core::TrajectoryDb &queries,
                             const Config &config)
    : knn(make_knn(database, queries, config)), where_at(make_where_at(database, config)),
      range(make_range(database, config)), join(make_join(database, queries, config)),
      self_join(make_self_join(database, config)),
      clustering(make_clustering(database, self_join ? *self_join : join, config)) {}

evaluator::Workloads TestWorkloads::view() const {
    const query::JoinWorkload *self = nullptr;
    if (self_join)
        self = &*self_join;
    else if (join.mode == query::JoinMode::Self)
        self = &join;
    return {.knn = &knn,
            .where_at = where_at.queries,
            .range = range.queries,
            .join = &join,
            .clustering = clustering ? &*clustering : nullptr,
            .self_join = self};
}

} // namespace pipeline

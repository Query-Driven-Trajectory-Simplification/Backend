#include "backend/engine/pipeline/query_generation.hpp"
#include "backend/engine/io/frontend_export.hpp"

#include <iostream>

namespace pipeline {

QuerySet generate_query_set(const core::TrajectoryDb &trajectories, const Config &config,
                            std::size_t count) {
    QuerySet set{.heat_map = query::heat_map(trajectories), .params = config.range_queries};
    set.params.count = count;
    const query::RangeQuerySampler sampler(trajectories);
    set.range_queries = sampler.sample(config.seed, set.params);
    if (set.range_queries.size() < count) {
        std::cerr << "only " << set.range_queries.size() << " of " << count
                  << " boxes passed the results rule\n";
    }
    return set;
}

void write_query_set(const QuerySet &set, const Config &config,
                     const std::filesystem::path &output_dir) {
    const auto heat_path = output_dir / "heatmap.json";
    io::export_heat_map(set.heat_map, heat_path);
    std::cout << set.heat_map.size() << " heat map cells written to " << heat_path.string()
              << '\n';

    const auto queries_path = output_dir / "range_queries.json";
    io::export_range_queries(set.range_queries, config.seed, set.params, queries_path);
    std::cout << set.range_queries.size() << " range queries (" << set.params.box_km << " km, "
              << set.params.window_seconds / 86400.0 << " days) written to "
              << queries_path.string() << '\n';
}

} // namespace pipeline

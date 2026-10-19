#include "backend/engine/pipeline/query_generation.hpp"
#include "backend/engine/io/frontend_export.hpp"
#include "backend/engine/query/heat_map.hpp"
#include "backend/engine/query/range_query.hpp"

#include <iostream>
#include <vector>

namespace pipeline {

void generate_queries(const core::TrajectoryDb &trajectories, const Config &config) {
    const auto heat = query::heat_map(trajectories);
    const auto heat_path = config.output_dir / "heatmap.json";
    io::export_heat_map(heat, heat_path);
    std::cout << heat.size() << " heat map cells written to " << heat_path.string() << '\n';

    const query::RangeQuerySampler sampler(trajectories);
    std::vector<io::SeedRangeQueries> seeds;
    for (const auto seed : config.seeds) {
        auto boxes = sampler.sample(seed, config.range_queries);
        if (boxes.size() < config.range_queries.count) {
            std::cerr << "seed " << seed << ": only " << boxes.size() << " of "
                      << config.range_queries.count << " boxes passed the results rule\n";
        }
        seeds.push_back({seed, std::move(boxes)});
    }
    const auto queries_path = config.output_dir / "range_queries.json";
    io::export_range_queries(seeds, config.range_queries, queries_path);
    std::cout << seeds.size() << " seeds x " << config.range_queries.count << " range queries ("
              << config.range_queries.box_km << " km, "
              << config.range_queries.window_seconds / 86400.0 << " days) written to "
              << queries_path.string() << '\n';
}

} // namespace pipeline

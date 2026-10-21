#include "backend/engine/pipeline/query_generation.hpp"
#include "backend/engine/io/frontend_export.hpp"
#include "backend/engine/query/heat_map.hpp"
#include "backend/engine/query/range_query.hpp"

#include <iostream>

namespace pipeline {

void generate_heat_map(const core::TrajectoryDb &trajectories, const Config &config) {
    const auto heat = query::heat_map(trajectories);
    const auto heat_path = config.output_dir / "heatmap.json";
    io::export_heat_map(heat, heat_path);
    std::cout << heat.size() << " heat map cells written to " << heat_path.string() << '\n';
}

void generate_queries(const core::TrajectoryDb &trajectories, const Config &config) {
    const query::RangeQuerySampler sampler(trajectories);
    const auto boxes = sampler.sample(config.seed, config.range_queries);
    if (boxes.size() < config.range_queries.count) {
        std::cerr << "only " << boxes.size() << " of " << config.range_queries.count
                  << " boxes passed the results rule\n";
    }
    const auto queries_path = config.output_dir / "range_queries.json";
    io::export_range_queries(boxes, config.seed, config.range_queries, queries_path);
    std::cout << boxes.size() << " range queries ("
              << config.range_queries.box_km << " km, "
              << config.range_queries.window_seconds / 86400.0 << " days) written to "
              << queries_path.string() << '\n';
}

} // namespace pipeline

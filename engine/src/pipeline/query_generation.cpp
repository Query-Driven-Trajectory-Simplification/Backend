#include "backend/engine/pipeline/query_generation.hpp"
#include "backend/engine/io/frontend_export.hpp"
#include "backend/engine/query/box_query.hpp"
#include "backend/engine/query/range_query.hpp"

#include <iostream>
#include <vector>

namespace pipeline {

void generate_queries(const core::TrajectoryDb &trajectories, const Config &config) {
    const auto heat = query::heat_map(trajectories);
    const auto heat_path = config.output_dir / "heatmap.json";
    io::export_heat_map(heat, heat_path);
    std::cout << heat.size() << " heat map cells written to " << heat_path.string() << '\n';

    const query::BoxQuerySampler sampler(trajectories);
    std::vector<io::SeedBoxQueries> box_seeds;
    for (const auto seed : config.seeds) {
        auto boxes = sampler.sample(seed, config.box_queries);
        if (boxes.size() < config.box_queries.count) {
            std::cerr << "seed " << seed << ": only " << boxes.size() << " of "
                      << config.box_queries.count << " boxes passed the results rule\n";
        }
        box_seeds.push_back({seed, std::move(boxes)});
    }
    const auto boxes_path = config.output_dir / "box_queries.json";
    io::export_box_queries(box_seeds, config.box_queries, boxes_path);
    std::cout << box_seeds.size() << " seeds x " << config.box_queries.count << " box queries ("
              << config.box_queries.box_km << " km, " << config.box_queries.window_seconds / 86400.0
              << " days) written to " << boxes_path.string() << '\n';
}

}

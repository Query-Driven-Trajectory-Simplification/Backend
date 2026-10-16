#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/io/frontend_export.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/query/range_query.hpp"

#include <iostream>
#include <set>
#include <vector>

namespace pipeline {

void run(const Config &config) {
    core::TrajectoryDb trajectories = io::load_trajectories(config.path);
    std::cout << trajectories.size() << " trips, " << trajectories.num_points() << " points\n";

    const auto heat = query::heat_map(trajectories);
    const auto heat_path = config.output_dir / "heatmap.json";
    io::export_heat_map(heat, heat_path);
    std::cout << heat.size() << " heat map cells written to " << heat_path.string() << '\n';

    std::vector<io::SeedQueries> seeds;
    std::set<query::Cell> distinct;
    for (const auto seed : config.seeds) {
        auto queries = query::sample_queries(heat, seed);
        for (const auto &q : queries) {
            distinct.insert(q.cell);
        }
        seeds.push_back({seed, std::move(queries)});
    }
    const auto queries_path = config.output_dir / "range_queries.json";
    io::export_range_queries(seeds, query::kMinQueryWeight, queries_path);
    std::cout << seeds.size() << " seeds x " << query::kQueriesPerSeed << " range queries ("
              << distinct.size() << " distinct cells) written to " << queries_path.string()
              << '\n';
}

} // namespace pipeline

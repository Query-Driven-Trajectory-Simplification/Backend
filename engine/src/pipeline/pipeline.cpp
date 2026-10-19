#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/io/frontend_export.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/query/box_query.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/selection/selector.hpp"

#include <iostream>
#include <set>
#include <vector>

namespace pipeline {

void run(const Config &config) {
    core::TrajectoryDb trajectories = io::load_trajectories(config.path);
    std::cout << trajectories.size() << " trajectories, " << trajectories.num_points()
              << " points\n";

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
              << distinct.size() << " distinct cells) written to " << queries_path.string() << '\n';

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

    std::cout << trajectories.trajectory(3).size() << std::endl;

    core::TrajectoryView trajectory = trajectories.trajectory(3);

    auto kept_indices = selection::simplify_trajectory(
        trajectory,
        selection::AlgorithmType::RDP,
        10
    );

    std::cout << "Simplified down to " << kept_indices.size() << " points. Indices:\n";
    for (std::size_t idx : kept_indices) {
        std::cout << "  index: " << idx
                  << " -> (x: " << trajectory.x[idx]
                  << ", y: " << trajectory.y[idx] << ")\n";
    }

    std::size_t total_budget = trajectories.num_points() / 2;
    auto budgets = allocation::create_trajectory_budgets(
        trajectories, allocation::AlgorithmType::kSizeProportional, total_budget);
}

} // namespace pipeline

#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/io/selection_file.hpp"
#include "backend/engine/pipeline/common.hpp"
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/query/range_query.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr std::size_t kAdjustBoxes = 100;
} // namespace

int main(int argc, char *argv[]) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: export_db <config.json> [adjust_seed=4242]\n";
        return 1;
    }

    try {
        const auto config = pipeline::load_config(argv[1]);
        const std::uint64_t adjust_seed = argc == 3 ? std::stoull(argv[2]) : 4242;
        if (adjust_seed == config.seed) {
            throw std::invalid_argument(
                "export_db: adjust_seed must differ from the config's seed, or MLSimp would see "
                "the evaluation's range queries");
        }

        const auto [database, queries] = pipeline::load_split(config);
        io::export_database(config.selection_dir, database);
        for (const double share : config.budget_shares) {
            const auto budgets = allocation::create_trajectory_budgets(
                database, allocation::AllocatorType::SizeProportional,
                allocation::total_budget(database, share));
            io::export_budgets(config.selection_dir, std::format("{:g}pct", share * 100), budgets);
        }

        auto range = config.range;
        range.query_count = kAdjustBoxes;
        const auto adjust = query::generate_range_queries(database, range, adjust_seed);
        const auto hit_count =
            io::export_range_hits(config.selection_dir, database, adjust.queries);

        std::cout << database.size() << " trajectories, " << database.num_points()
                  << " points written to " << config.selection_dir.string() << '\n'
                  << adjust.queries.size() << " range queries (seed " << adjust_seed << ") cover "
                  << hit_count << " points\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

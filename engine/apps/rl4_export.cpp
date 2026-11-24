// Writes what RL4QDTS trains on for one seed (see rl4qdts/README.md): the database's trajectory
// ids (database_ids.csv, for export_rl4.py --ids) and the teacher's practice range queries
// (range_practice.parquet, for rl4_train.py --range-queries). The test workloads are not written:
// RL4 is scored on them by the engine, so it must not train on them.
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/io/workload_export.hpp"
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/split/split.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    if (argc != 3 && !(argc == 5 && std::string(argv[3]) == "--seed")) {
        std::cerr << "usage: rl4_export <config.json> <out_dir> [--seed N]\n";
        return 1;
    }
    try {
        auto config = pipeline::load_config(argv[1]);
        if (argc == 5)
            config.seed = std::stoull(argv[4]);
        const std::filesystem::path out = argv[2];

        // The same database as run.exe and teacher_eval for this seed
        const auto all = io::load_trajectories(config.data_path);
        const auto parts = split::split(all, config.split, config.seed);
        const auto database = core::subset(all, parts.database);

        // The teacher's practice workload (teacher/workloads.cpp)
        const auto range =
            query::generate_range_queries(database, pipeline::teacher_workload_config(config).range,
                                          config.seed + teacher::kRangeSeedOffset);

        io::export_database_ids(out, database);
        io::export_range_boxes(out / "range_practice.parquet", database, range.queries);
        std::cout << "seed " << config.seed << ": " << database.size() << " trajectories, "
                  << range.queries.size() << " practice range queries written to " << out.string()
                  << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

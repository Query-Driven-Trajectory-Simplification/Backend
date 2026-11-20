#include "backend/engine/pipeline/common.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/split/split.hpp"

namespace pipeline {

Databases load_split(const Config &config) {
    const core::TrajectoryDb all = io::load_trajectories(config.data_path);
    const auto parts = split::split(all, config.split, config.seed);
    return {.database = core::subset(all, parts.database),
            .queries = core::subset(all, parts.queries)};
}

} // namespace pipeline

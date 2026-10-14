#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/core/types.hpp"
#include "backend/engine/io/parquet_loader.hpp"

namespace pipeline {

void run(const Config& config) {
    io::load_trajectories(config.path);
}

}
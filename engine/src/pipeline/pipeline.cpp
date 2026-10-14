#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/io/parquet_loader.hpp"

namespace pipeline {

void run(const Config& config) {
    auto data  = io::read_parquet(config.path);
}

}
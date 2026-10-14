#include <optional>
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/pipeline/pipeline.hpp"

std::optional<pipeline::Config> parse_args(int argc, char* argv[]) {
    if (argc != 2) {
        return std::nullopt;
    }

    return pipeline::Config{.path = argv[1]};
}

int main(int argc, char* argv[]) {
    auto config = parse_args(argc, argv);
    if (!config) return 1;

    pipeline::run(*config);
}
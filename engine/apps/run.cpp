#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/pipeline/pipeline.hpp"
#include <iostream>
#include <optional>

std::optional<pipeline::Config> parse_args(int argc, char *argv[]) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: " << argv[0] << " <points.parquet> [output_dir]\n";
        return std::nullopt;
    }

    pipeline::Config config{.path = argv[1]};
    if (argc == 3) {
        config.output_dir = argv[2];
    }
    return config;
}

int main(int argc, char *argv[]) {
    auto config = parse_args(argc, argv);
    if (!config)
        return 1;

    try {
        pipeline::run(*config);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

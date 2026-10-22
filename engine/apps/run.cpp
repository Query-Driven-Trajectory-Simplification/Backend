#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/pipeline/pipeline.hpp"
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

std::optional<pipeline::Config> parse_args(int argc, char *argv[]) {
    std::vector<std::string_view> positional;
    bool write_json = false;
    std::optional<std::uint64_t> seed;
    std::optional<std::uint64_t> eval_seed;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--json") {
            write_json = true;
        } else if (arg == "--seed" || arg == "--eval-seed") {
            if (i + 1 >= argc) {
                std::cerr << arg << " needs a value\n";
                return std::nullopt;
            }
            (arg == "--seed" ? seed : eval_seed) = std::stoull(argv[++i]);
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.empty() || positional.size() > 3) {
        std::cerr << "usage: " << argv[0]
                  << " <parquet> [train_queries [test_queries]] [--json] [--seed N] "
                     "[--eval-seed N]\n";
        return std::nullopt;
    }

    pipeline::Config config{.path = positional[0], .write_json = write_json};
    if (seed) {
        config.seed = *seed;
    }
    if (eval_seed) {
        config.eval_seed = *eval_seed;
    }
    if (positional.size() >= 2) {
        config.train_queries = std::stoull(std::string(positional[1]));
    }
    if (positional.size() == 3) {
        config.test_queries = std::stoull(std::string(positional[2]));
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

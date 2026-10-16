#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace pipeline {

struct Config {
    std::filesystem::path path;
    std::filesystem::path output_dir = "output";
    // One range query set per seed, so results can be reported as mean and spread over seeds.
    std::vector<std::uint64_t> seeds = {1, 2, 3, 4, 5};
};

} // namespace pipeline

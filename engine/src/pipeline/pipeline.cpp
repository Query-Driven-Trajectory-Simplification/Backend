#include "backend/engine/pipeline/pipeline.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/split/split.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string_view>

namespace pipeline {

namespace {

constexpr double kQueryPercentage = 0.15;
constexpr std::uint64_t kSplitSeed = 1;

struct Databases {
    core::TrajectoryDb database;
    core::TrajectoryDb queries;
};

Databases load_split(const std::filesystem::path &path) {
    const core::TrajectoryDb all = io::load_trajectories(path);
    const auto parts = split::split(all, split::SplitType::Random, kQueryPercentage, kSplitSeed);
    return {.database = core::subset(all, parts.database),
            .queries = core::subset(all, parts.queries)};
}

void print_size(std::string_view name, const core::TrajectoryDb &trajectories) {
    std::cout << name << ": " << trajectories.size() << " trajectories, "
              << trajectories.num_points() << " points\n";
}

} // namespace

void run(const Config &config) {
    const auto [database, queries] = load_split(config.path);
    print_size("database", database);
    print_size("queries", queries);
}

} // namespace pipeline
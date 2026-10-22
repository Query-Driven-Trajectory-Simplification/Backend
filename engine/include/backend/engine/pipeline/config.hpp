#pragma once

#include "backend/engine/query/range_query.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace pipeline {

struct Config {
    std::filesystem::path path;
    std::filesystem::path output_dir = "output";
    std::uint64_t seed = 1;
    // Seed for the range queries the evaluator scores with. Kept apart from seed, so the default
    // run does not evaluate on the same queries it generates for the database (train) side.
    std::uint64_t eval_seed = 2;
    // Number of range queries (and the heat map) generated for the database (train) and queries
    // (test) side. 0 skips generating for that side.
    // TODO: Can be changed or made smarter in the future.
    std::size_t train_queries = 500;
    std::size_t test_queries = 500;
    // Also write the heat maps and range queries to output_dir/{database,queries} as JSON.
    bool write_json = false;
    query::RangeQueryParams range_queries;
};

} // namespace pipeline

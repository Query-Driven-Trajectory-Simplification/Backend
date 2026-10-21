#pragma once

#include "backend/engine/query/range_query.hpp"

#include <cstdint>
#include <filesystem>

namespace pipeline {

struct Config {
    std::filesystem::path path;
    std::filesystem::path output_dir = "output";
    std::uint64_t seed = 1;
    // range_queries.count is the number of queries in the single generated set.
    query::RangeQueryParams range_queries;
};

} // namespace pipeline

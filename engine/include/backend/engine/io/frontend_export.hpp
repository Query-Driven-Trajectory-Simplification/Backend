#pragma once

#include "backend/engine/query/box_query.hpp"
#include "backend/engine/query/range_query.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace io {

// Every cell is written as
// {"lat_idx", "lon_idx", "t_idx", "min_lat", "max_lat", "min_lon", "max_lon",
//  "t_start", "t_end", "weight"}, bounds half-open [min, max), times Unix seconds UTC.

// {"cell_size": {"lat", "lon", "seconds"}, "cells": [cell, ...]},
// cells sorted by week, then lat, then lon.
void export_heat_map(const query::HeatMap &heat_map, const std::filesystem::path &path);

struct SeedQueries {
    std::uint64_t seed;
    std::vector<query::RangeQuery> queries;
};

// {"min_weight", "queries_per_seed", "seeds": [{"seed", "queries": [cell, ...]}, ...]},
// queries in the order they were drawn.
void export_range_queries(std::span<const SeedQueries> seeds, std::uint32_t min_weight,
                          const std::filesystem::path &path);

struct SeedBoxQueries {
    std::uint64_t seed;
    std::vector<query::BoxQuery> queries;
};

void export_box_queries(std::span<const SeedBoxQueries> seeds, const query::BoxQueryParams &params,
                        const std::filesystem::path &path);

} // namespace io

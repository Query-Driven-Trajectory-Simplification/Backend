#pragma once

#include "backend/engine/query/heat_map.hpp"
#include "backend/engine/query/range_query.hpp"

#include <cstdint>
#include <filesystem>
#include <span>

namespace io {

// Every cell is written as
// {"lat_idx", "lon_idx", "t_idx", "min_lat", "max_lat", "min_lon", "max_lon",
//  "t_start", "t_end", "weight"}, bounds half-open [min, max), times Unix seconds UTC.

// {"cell_size": {"lat", "lon", "seconds"}, "cells": [cell, ...]},
// cells sorted by week, then lat, then lon.
void export_heat_map(const query::HeatMap &heat_map, const std::filesystem::path &path);

void export_range_queries(std::span<const query::RangeQuery> queries, std::uint64_t seed,
                          const query::RangeQueryParams &params, const std::filesystem::path &path);

} // namespace io

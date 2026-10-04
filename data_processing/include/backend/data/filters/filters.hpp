#pragma once

#include <array>
#include <cstddef>
#include <ostream>
#include <string_view>
#include <vector>

#include "backend/data/readers/geolife.hpp"

namespace filters {

// Points and segments that remain after one stage of the pipeline.
struct StageStats {
    std::string_view name;
    std::size_t points = 0;
    std::size_t segments = 0;
};

// Totals per stage, accumulated over every trajectory passed to apply_filters.
struct FilterStats {
    std::array<StageStats, 5> stages = {{
        {"input"},
        {"bounding box"},
        {"time gap"},
        {"stay points"},
        {"min trip"},
    }};
};

// Runs all filters on one trajectory and returns the segments that remain.
std::vector<geolife::Segment> apply_filters(const geolife::Segment& points, FilterStats& stats);

// Prints one row per stage, compared against the input and the previous stage.
void print_stats(std::ostream& out, const FilterStats& stats);

}

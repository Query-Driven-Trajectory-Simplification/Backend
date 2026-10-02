#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace geolife {

struct Point {
    double lat;
    double lon;
    double alt_m;
    std::int64_t t;
};

std::optional<Point> parse_plt_line(const std::string& line);
std::vector<std::filesystem::path> collect_plt_files(const std::filesystem::path& root);

}

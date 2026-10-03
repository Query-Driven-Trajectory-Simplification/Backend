#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace geolife {

struct Point {
    double lat;
    double lon;
    std::int64_t t;
};

using Segment = std::vector<Point>;

std::expected<Point, std::string> parse_plt_line(const std::string& line);

std::expected<Segment, std::string> read_plt_file(const std::filesystem::path& path);

std::vector<std::filesystem::path> collect_plt_files(const std::filesystem::path& root);

}

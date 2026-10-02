#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace geolife {

struct Point {
    double lat;
    double lon;
    double alt_m;
    std::int64_t t;
};

struct PltFile {
    std::vector<Point> points;
    std::size_t bad_lines = 0;
};

std::expected<Point, std::string> parse_plt_line(const std::string& line);

std::expected<PltFile, std::string> read_plt_file(const std::filesystem::path& path);

std::vector<std::filesystem::path> collect_plt_files(const std::filesystem::path& root);

}

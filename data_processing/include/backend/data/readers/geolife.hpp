#pragma once

#include <cstddef>
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

using Segment = std::vector<Point>;

struct PltFile {
    std::vector<Point> points;
    std::size_t bad_lines = 0;
};

std::optional<Point> parse_plt_line(const std::string& line);

std::optional<PltFile> read_plt_file(const std::filesystem::path& path);

std::vector<std::filesystem::path> collect_plt_files(const std::filesystem::path& root);

}

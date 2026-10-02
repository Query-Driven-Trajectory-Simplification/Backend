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

struct PltFile {
    std::vector<Point> points;
    std::size_t bad_lines = 0;
};

std::optional<Point> parse_plt_line(const std::string& line);

// Reads one .plt file: skips the header and parses every line.
// Returns nullopt if the file cannot be opened.
std::optional<PltFile> read_plt_file(const std::filesystem::path& path);

// All .plt files under root, sorted so indices are stable across machines.
std::vector<std::filesystem::path> collect_plt_files(const std::filesystem::path& root);

}

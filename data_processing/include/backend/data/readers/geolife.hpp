#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace geolife {

struct Point {
    double lat;
    double lon;
    std::int64_t t;
};

using Segment = std::vector<Point>;

// Seconds since 1970-01-01 for a date and time read as UTC, or nullopt if it doesn't exist.
std::optional<std::int64_t> to_unix_seconds(int y, int mo, int d, int h, int mi, int s);

std::expected<Point, std::string> parse_plt_line(const std::string &line);

std::expected<Segment, std::string> read_plt_file(const std::filesystem::path &path);

std::vector<std::filesystem::path> collect_plt_files(const std::filesystem::path &root);

} // namespace geolife

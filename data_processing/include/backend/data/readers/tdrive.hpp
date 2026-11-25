#pragma once

#include "backend/data/readers/geolife.hpp"

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

// T-Drive taxi logs: one .txt file per taxi, one point per line.
// Points use geolife::Point, so the same filters and writer apply.
namespace tdrive {

// Parses "taxi id,date time,longitude,latitude", e.g. "1,2008-02-02 15:36:08,116.51172,39.92123".
// Longitude comes before latitude. The time is kept as written: the user guide gives no time zone.
std::expected<geolife::Point, std::string> parse_line(const std::string &line);

// Every point of one taxi file, in file order. Unreadable lines are reported and skipped.
std::expected<geolife::Segment, std::string> read_file(const std::filesystem::path &path);

// Every .txt file under root, sorted by path.
std::vector<std::filesystem::path> collect_files(const std::filesystem::path &root);

} // namespace tdrive

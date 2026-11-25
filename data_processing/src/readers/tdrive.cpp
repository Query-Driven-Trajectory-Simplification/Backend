#include "backend/data/readers/tdrive.hpp"

#include <algorithm>
#include <cstdio>
#include <format>
#include <fstream>
#include <iostream>

namespace tdrive {

std::expected<geolife::Point, std::string> parse_line(const std::string &line) {
    geolife::Point point{};
    int year, month, day;
    int hour, minute, second;

    // taxi id (skipped), date time, longitude, latitude
    const int fields = std::sscanf(line.c_str(), "%*d,%d-%d-%d %d:%d:%d,%lf,%lf", &year, &month,
                                   &day, &hour, &minute, &second, &point.lon, &point.lat);
    if (fields != 8) {
        return std::unexpected(
            std::format("expected 8 fields, could only read {}", fields < 0 ? 0 : fields));
    }

    const auto t = geolife::to_unix_seconds(year, month, day, hour, minute, second);
    if (!t) {
        return std::unexpected(std::format("invalid date/time {:04}-{:02}-{:02} {:02}:{:02}:{:02}",
                                           year, month, day, hour, minute, second));
    }
    point.t = *t;
    return point;
}

std::expected<geolife::Segment, std::string> read_file(const std::filesystem::path &path) {
    std::ifstream in(path);
    if (!in)
        return std::unexpected("cannot open " + path.string());

    geolife::Segment segment;
    std::string line;
    for (int line_no = 1; std::getline(in, line); ++line_no) {
        auto p = parse_line(line);
        if (!p) {
            std::cerr << path.string() << ':' << line_no << ": " << p.error() << '\n';
            continue;
        }
        segment.push_back(*p);
    }
    return segment;
}

std::vector<std::filesystem::path> collect_files(const std::filesystem::path &root) {
    std::vector<std::filesystem::path> files;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace tdrive

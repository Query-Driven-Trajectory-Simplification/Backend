#include "backend/data/readers/geolife.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <format>
#include <expected>
#include <fstream>
#include <iostream>
#include <optional>

namespace geolife {

namespace {

constexpr int kHeaderLines = 6;

std::optional<std::int64_t> to_unix_seconds(int y, int mo, int d, int h, int mi, int s) {
    using namespace std::chrono;
    const year_month_day date = year{y} / mo / d;
    if (!date.ok() || h < 0 || h > 23 || mi < 0 || mi > 59 || s < 0 || s > 59)
        return std::nullopt;
    return (sys_days{date} + hours{h} + minutes{mi} + seconds{s}).time_since_epoch().count();
}

}

std::expected<Point, std::string> parse_plt_line(const std::string& line) {
    Point point{};

    int year, month, day;
    int hour, minute, second;

    // lat, lon, 0, altitude (skipped), days since 1899 (skipped), date, time
    const int fields = std::sscanf(line.c_str(), "%lf,%lf,0,%*lf,%*f,%d-%d-%d,%d:%d:%d",
            &point.lat, &point.lon,
            &year, &month, &day, &hour, &minute, &second);
    if (fields != 8) {
        return std::unexpected(std::format("expected 8 fields, could only read {}", fields < 0 ? 0 : fields));
    }

    // datetime to timestamp converter, UTC
    const auto t = to_unix_seconds(year, month, day, hour, minute, second);
    if (!t) {
        return std::unexpected(std::format("invalid date/time {:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            year, month, day, hour, minute, second));
    }
    point.t = *t;
    return point;
}

std::expected<Segment, std::string> read_plt_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return std::unexpected("cannot open " + path.string());

    Segment segment;
    std::string line;

    int line_no = 0;

    for (; line_no < kHeaderLines && std::getline(in, line); ++line_no) {}

    while (std::getline(in, line)) {
        ++line_no;
        auto p = parse_plt_line(line);
        if (!p) {
            std::cerr << path.string() << ':' << line_no << ": " << p.error() << '\n';
            continue;
        }
        segment.push_back(*p);
    }
    return segment;
}

std::vector<std::filesystem::path> collect_plt_files(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".plt")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

}

#include "backend/data/readers/geolife.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
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

std::optional<Point> parse_plt_line(const std::string& line) {
    Point point{};

    int year, month, day;
    int hour, minute, second;

    // lat, lon, 0, altitude (feet), days since 1899 (skipped), date, time
    if (std::sscanf(line.c_str(), "%lf,%lf,0,%lf,%*f,%d-%d-%d,%d:%d:%d",
            &point.lat, &point.lon, &point.alt_m,
            &year, &month, &day, &hour, &minute, &second) != 9) {
        return std::nullopt;
    }

    // -777 means unknown.
    if (point.alt_m == -777) {
        point.alt_m = std::numeric_limits<double>::quiet_NaN();
    } else {
        point.alt_m *= 0.3048; // feet to meter conversion
    }

    // datetime to timestamp converter, UTC
    const auto t = to_unix_seconds(year, month, day, hour, minute, second);
    if (!t) return std::nullopt;
    point.t = *t;
    return point;
}

std::optional<PltFile> read_plt_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return std::nullopt;

    PltFile file;
    std::string line;

    for (int i = 0; i < kHeaderLines && std::getline(in, line); ++i) {}

    while (std::getline(in, line)) {
        if (auto p = parse_plt_line(line)) {
            file.points.push_back(*p);
        } else {
            ++file.bad_lines;
        }
    }
    return file;
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

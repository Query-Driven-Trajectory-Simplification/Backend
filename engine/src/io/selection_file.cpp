#include "backend/engine/io/selection_file.hpp"
#include "backend/engine/core/parallel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace io {
namespace {

template <typename T> void write_array(const std::filesystem::path &path, std::span<const T> data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size_bytes()));
    if (!out) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

} // namespace

void export_database(const std::filesystem::path &dir, const core::TrajectoryDb &database) {
    std::filesystem::create_directories(dir);
    std::vector<std::int32_t> ids;
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int64_t> t;
    std::vector<double> lat;
    std::vector<double> lon;
    ids.reserve(database.size());
    offsets.reserve(database.size() + 1);
    t.reserve(database.num_points());
    lat.reserve(database.num_points());
    lon.reserve(database.num_points());
    for (std::size_t i = 0; i < database.size(); ++i) {
        const auto trajectory = database.trajectory(i);
        ids.push_back(trajectory.trajectory_id);
        t.insert(t.end(), trajectory.t.begin(), trajectory.t.end());
        lat.insert(lat.end(), trajectory.lat.begin(), trajectory.lat.end());
        lon.insert(lon.end(), trajectory.lon.begin(), trajectory.lon.end());
        offsets.push_back(t.size());
    }
    write_array<std::int32_t>(dir / "ids.bin", ids);
    write_array<std::uint64_t>(dir / "offsets.bin", offsets);
    write_array<std::int64_t>(dir / "t.bin", t);
    write_array<double>(dir / "lat.bin", lat);
    write_array<double>(dir / "lon.bin", lon);
}

void export_budgets(const std::filesystem::path &dir, std::string_view name,
                    std::span<const std::size_t> budgets) {
    std::filesystem::create_directories(dir);
    const std::vector<std::uint64_t> values(budgets.begin(), budgets.end());
    write_array<std::uint64_t>(dir / ("budgets_" + std::string(name) + ".bin"), values);
}

std::size_t export_range_hits(const std::filesystem::path &dir, const core::TrajectoryDb &database,
                              std::span<const query::RangeQuery> queries) {
    std::filesystem::create_directories(dir);
    std::vector<std::uint8_t> hits(database.num_points(), 0);
    std::vector<std::size_t> starts(database.size() + 1, 0);
    for (std::size_t i = 0; i < database.size(); ++i) {
        starts[i + 1] = starts[i] + database.trajectory(i).size();
    }

    core::parallel_for(database.size(), [&](std::size_t i) {
        const auto trajectory = database.trajectory(i);
        const auto &b = trajectory.bbox;
        for (const auto &query : queries) {
            const auto &box = query.box;
            if (b.x_max < box.x_min || b.x_min > box.x_max || b.y_max < box.y_min ||
                b.y_min > box.y_max || trajectory.t.back() < box.t_min ||
                trajectory.t.front() > box.t_max) {
                continue;
            }
            const auto first = std::ranges::lower_bound(trajectory.t, box.t_min);
            for (auto k = static_cast<std::size_t>(first - trajectory.t.begin());
                 k < trajectory.size() && trajectory.t[k] <= box.t_max; ++k) {
                if (trajectory.x[k] >= box.x_min && trajectory.x[k] <= box.x_max &&
                    trajectory.y[k] >= box.y_min && trajectory.y[k] <= box.y_max) {
                    hits[starts[i] + k] = 1;
                }
            }
        }
    });
    write_array<std::uint8_t>(dir / "range_hits.bin", hits);
    return static_cast<std::size_t>(std::ranges::count(hits, std::uint8_t{1}));
}

core::KeptSets read_selection(const std::filesystem::path &path,
                              const core::TrajectoryDb &database) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::invalid_argument("read_selection: cannot open " + path.string());
    }
    const auto bytes = static_cast<std::size_t>(in.tellg());
    if (bytes % sizeof(std::uint64_t) != 0) {
        throw std::invalid_argument(
            std::format("read_selection: {} holds {} bytes, not a whole number of uint64 indices",
                        path.string(), bytes));
    }
    std::vector<std::uint64_t> kept(bytes / sizeof(std::uint64_t));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(kept.data()), static_cast<std::streamsize>(bytes));

    const auto fail = [&](std::string_view reason) {
        throw std::invalid_argument(std::format(
            "read_selection: {}: {}. Was it made from the same database?", path.string(), reason));
    };
    for (std::size_t k = 1; k < kept.size(); ++k) {
        if (kept[k] <= kept[k - 1]) {
            fail("indices must be strictly increasing");
        }
    }
    if (!kept.empty() && kept.back() >= database.num_points()) {
        fail(std::format("index {} is past the database's {} points", kept.back(),
                         database.num_points()));
    }

    core::KeptSets sets(database.size());
    std::size_t next = 0;
    std::uint64_t begin = 0;
    for (std::size_t i = 0; i < database.size(); ++i) {
        const std::uint64_t size = database.trajectory(i).size();
        const std::uint64_t end = begin + size;
        while (next < kept.size() && kept[next] < end) {
            sets[i].push_back(static_cast<std::size_t>(kept[next] - begin));
            ++next;
        }
        if (size > 0 && (sets[i].empty() || sets[i].front() != 0 || sets[i].back() != size - 1)) {
            fail(std::format("trajectory {} must keep its first and last point",
                             database.trajectory(i).trajectory_id));
        }
        begin = end;
    }
    return sets;
}

} // namespace io

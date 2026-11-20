#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/io/selection_file.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

core::TrajectoryDb MakeDb(const std::vector<std::size_t> &sizes) {
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        for (std::size_t k = 0; k < sizes[i]; ++k) {
            t.push_back(static_cast<std::int64_t>(k));
        }
        offsets.push_back(t.size());
        ids.push_back(static_cast<std::int32_t>(i));
    }
    std::vector<double> lat(t.size());
    std::vector<double> lon(t.size());
    return core::TrajectoryDb(offsets, ids, t, lat, lon);
}

std::filesystem::path WriteIndices(const std::string &name,
                                   const std::vector<std::uint64_t> &indices) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(indices.data()),
              static_cast<std::streamsize>(indices.size() * sizeof(std::uint64_t)));
    return path;
}

} // namespace

TEST(SelectionFile, ReadsGlobalIndicesIntoTrajectories) {
    const auto db = MakeDb({5, 3});
    const auto kept = io::read_selection(WriteIndices("selection_ok.bin", {0, 2, 4, 5, 7}), db);
    EXPECT_EQ(kept, (core::KeptSets{{0, 2, 4}, {0, 2}}));
    const auto uneven =
        io::read_selection(WriteIndices("selection_uneven.bin", {0, 4, 5, 6, 7}), db);
    EXPECT_EQ(uneven, (core::KeptSets{{0, 4}, {0, 1, 2}}));
}

TEST(SelectionFile, RejectsMissingEndpointsUnsortedAndOutOfRangeIndices) {
    const auto db = MakeDb({5, 3});
    EXPECT_THROW(io::read_selection(WriteIndices("selection_no_end.bin", {0, 2, 3, 5, 7}), db),
                 std::invalid_argument);
    EXPECT_THROW(io::read_selection(WriteIndices("selection_dup.bin", {0, 4, 4, 5, 7}), db),
                 std::invalid_argument);
    EXPECT_THROW(io::read_selection(WriteIndices("selection_past.bin", {0, 4, 5, 7, 8}), db),
                 std::invalid_argument);
    const auto partial = std::filesystem::temp_directory_path() / "selection_partial.bin";
    std::ofstream(partial, std::ios::binary).write("abc", 3);
    EXPECT_THROW(io::read_selection(partial, db), std::invalid_argument);
    EXPECT_THROW(io::read_selection("missing_selection.bin", db), std::invalid_argument);
}

TEST(SelectionFile, MarksPointsInsideRangeBoxes) {
    const auto db = MakeDb({5, 3});
    const auto dir = std::filesystem::temp_directory_path() / "selection_hits";
    const std::vector<query::RangeQuery> queries{
        {.box =
             {.x_min = -1e12, .x_max = 1e12, .y_min = -1e12, .y_max = 1e12, .t_min = 1, .t_max = 2},
         .reference = {}}};
    EXPECT_EQ(io::export_range_hits(dir / "range_hits.bin", db, queries), 4u);
    std::ifstream in(dir / "range_hits.bin", std::ios::binary);
    std::vector<char> hits(8);
    in.read(hits.data(), 8);
    EXPECT_EQ(hits, (std::vector<char>{0, 1, 1, 0, 0, 0, 1, 1}));
    EXPECT_EQ(in.peek(), std::char_traits<char>::eof());
}

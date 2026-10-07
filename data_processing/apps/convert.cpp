#include "backend/data/readers/geolife.hpp"
#include "backend/data/filters/filters.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

namespace fs = std::filesystem;

namespace {

struct Args {
    fs::path input;
    fs::path output;
};

std::optional<Args> parse_args(int argc, char* argv[]){
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <input_dir> <output_file>\n";
        return std::nullopt;
    }

    const fs::path input  = argv[1];
    const fs::path output = argv[2];

    if (!fs::is_directory(input)) {
        std::cerr << "not a directory: " << input << '\n';
        return std::nullopt;
    }

    return Args{input, output};
}

struct PointTableBuilder {
    arrow::Int32Builder  trajectory_id;
    arrow::Int64Builder  t;
    arrow::DoubleBuilder lat, lon;

    arrow::Status append(std::int32_t id, const geolife::Point& p) {
        ARROW_RETURN_NOT_OK(trajectory_id.Append(id));
        ARROW_RETURN_NOT_OK(t.Append(p.t));
        ARROW_RETURN_NOT_OK(lat.Append(p.lat));
        ARROW_RETURN_NOT_OK(lon.Append(p.lon));
        return arrow::Status::OK();
    }

    arrow::Result<std::shared_ptr<arrow::Table>> finish() {
        ARROW_ASSIGN_OR_RAISE(auto trajectory_id_arr, trajectory_id.Finish());
        ARROW_ASSIGN_OR_RAISE(auto t_arr,     t.Finish());
        ARROW_ASSIGN_OR_RAISE(auto lat_arr,   lat.Finish());
        ARROW_ASSIGN_OR_RAISE(auto lon_arr,   lon.Finish());

        auto schema = arrow::schema({
            arrow::field("trajectory_id", arrow::int32()),
            arrow::field("t",             arrow::int64()),  // UTC
            arrow::field("lat",           arrow::float64()),
            arrow::field("lon",           arrow::float64())
        });
        return arrow::Table::Make(
            schema,
            {trajectory_id_arr, t_arr, lat_arr, lon_arr});
    }
};

arrow::Status write_parquet(const arrow::Table& table, const fs::path& output) {
    if (output.has_parent_path()) {
        fs::create_directories(output.parent_path());
    }
    ARROW_ASSIGN_OR_RAISE(auto outfile, arrow::io::FileOutputStream::Open(output.string()));
    return parquet::arrow::WriteTable(table, arrow::default_memory_pool(), outfile);
}

arrow::Status convert(const fs::path& input, const fs::path& output) {
    const auto files = geolife::collect_plt_files(input);
    PointTableBuilder builder;
    std::size_t points = 0, stored_points = 0;
    std::size_t trajectory_id = 0;

    for (const auto& file : files) {
        auto segment = geolife::read_plt_file(file);
        if (!segment) return arrow::Status::IOError(segment.error());

        points += segment->size();
        auto filtered_segments = filters::apply_filters(std::move(*segment));
        for (const auto& filtered_segment: filtered_segments) {
            for (const auto& p : filtered_segment) {
                ARROW_RETURN_NOT_OK(builder.append(static_cast<std::int32_t>(trajectory_id), p));
            }
            stored_points += filtered_segment.size();
            trajectory_id++;
        }
    }

    std::cout << files.size() << " files, " << points << " input points\n"
              << stored_points << " points stored in " << trajectory_id << " trajectories\n";

    ARROW_ASSIGN_OR_RAISE(auto table, builder.finish());
    return write_parquet(*table, output);
}

}

int main(int argc, char* argv[]) {
    auto path_arguments = parse_args(argc, argv);
    if (!path_arguments) {
        return 1;
    }

    arrow::Status st = convert(path_arguments->input, path_arguments->output);
    if (!st.ok()) {
        std::cerr << st.ToString() << '\n';
        return 1;
    }
}

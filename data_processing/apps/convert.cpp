#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include "backend/data/readers/geolife.hpp"

namespace fs = std::filesystem;

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

arrow::Status convert(const fs::path& input, const fs::path& output) {
    arrow::Int32Builder  trajectory_id_b;
    arrow::Int64Builder  t_b;
    arrow::DoubleBuilder lat_b, lon_b, alt_b;

    const auto files = geolife::collect_plt_files(input);
    std::size_t points = 0, bad = 0;

    for (std::size_t trajectory_id = 0; trajectory_id < files.size(); ++trajectory_id) {
        if (trajectory_id % 100 == 0)
        {
            std::cout << "processing trajectory: " << trajectory_id << '\n';
        }
        std::ifstream in(files[trajectory_id]);
        if (!in) return arrow::Status::IOError("cannot open ", files[trajectory_id].string());

        std::string line;

        // skip the 6 header lines
        for (int i = 0; i < 6 && std::getline(in, line); ++i) {}

        while (std::getline(in, line)) {
            auto p = geolife::parse_plt_line(line);
            if (!p) {
                ++bad;
                continue;
            }

            ARROW_RETURN_NOT_OK(trajectory_id_b.Append(static_cast<std::int32_t>(trajectory_id)));
            ARROW_RETURN_NOT_OK(t_b.Append(p->t));
            ARROW_RETURN_NOT_OK(lat_b.Append(p->lat));
            ARROW_RETURN_NOT_OK(lon_b.Append(p->lon));
            if (std::isnan(p->alt_m)) {
                ARROW_RETURN_NOT_OK(alt_b.AppendNull());
            } else {
                ARROW_RETURN_NOT_OK(alt_b.Append(p->alt_m));
            }
            ++points;
        }
    }

    std::cout << files.size() << " files, " << points << " points, " << bad << " bad lines\n";

    ARROW_ASSIGN_OR_RAISE(auto trajectory_id_arr, trajectory_id_b.Finish());
    ARROW_ASSIGN_OR_RAISE(auto t_arr,   t_b.Finish());
    ARROW_ASSIGN_OR_RAISE(auto lat_arr, lat_b.Finish());
    ARROW_ASSIGN_OR_RAISE(auto lon_arr, lon_b.Finish());
    ARROW_ASSIGN_OR_RAISE(auto alt_arr, alt_b.Finish());

    auto schema = arrow::schema({
        arrow::field("trajectory_id", arrow::int32()),
        arrow::field("t",             arrow::int64()),
        arrow::field("lat",           arrow::float64()),
        arrow::field("lon",           arrow::float64()),
        arrow::field("alt_m",         arrow::float64()),
    });
    auto table = arrow::Table::Make(
        schema, 
        {trajectory_id_arr, t_arr, lat_arr, lon_arr, alt_arr});

    if (output.has_parent_path()) {
        fs::create_directories(output.parent_path());
    }
    ARROW_ASSIGN_OR_RAISE(auto outfile, arrow::io::FileOutputStream::Open(output.string()));
    ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile));

    return arrow::Status::OK();
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

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#include "backend/data/readers/geolife.hpp"

namespace fs = std::filesystem;

struct Args {
    fs::path input;
    fs::path output;
};

struct Row {
    std::size_t trajectory_id;
    std::int64_t t;
    double lat, lon, alt_m;
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

int main(int argc, char* argv[]) {
    auto path_arguments = parse_args(argc, argv);
    if (!path_arguments) {
        return 1;
    }

    const auto& input = path_arguments->input;
    const auto& output = path_arguments->output;

    const auto files = geolife::collect_plt_files(input);
    std::size_t points = 0, bad = 0;

    for (std::size_t trajectory_id = 0; trajectory_id < files.size(); ++trajectory_id) {
        std::ifstream in(files[trajectory_id]);
        std::string line;

        // skip the 6 header lines
        for (int i = 0; i < 6 && std::getline(in, line); ++i) {}

        while (std::getline(in, line)) {
            if (auto p = geolife::parse_plt_line(line)) {
                Row row{trajectory_id, p->t, p->lat, p->lon, p->alt_m};
                std::cout << trajectory_id << ' ' << p->t << ' ' << p->lat << ' ' << p->lon << ' ' << p->alt_m << '\n';
                ++points;
            } else {
                ++bad;
            }
        }
    }

    std::cout << files.size() << " files, " << points << " points, " << bad << " bad lines\n";
}
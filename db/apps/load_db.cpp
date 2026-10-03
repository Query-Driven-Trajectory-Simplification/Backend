#include "db/db.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " <points.parquet> <database_file> <dataset>\n";
        return 1;
    }

    const fs::path parquet  = argv[1];
    const fs::path database = argv[2];
    const std::string dataset = argv[3];

    if (!fs::is_regular_file(parquet)) {
        std::cerr << "not a file: " << parquet << '\n';
        return 1;
    }
    if (database.has_parent_path()) {
        fs::create_directories(database.parent_path());
    }

    auto db = db::Database::open(database);
    if (!db) {
        std::cerr << db.error() << '\n';
        return 1;
    }

    if (auto schema = db::create_schema(*db); !schema) {
        std::cerr << schema.error() << '\n';
        return 1;
    }

    auto points = db::load_points(*db, parquet, dataset);
    if (!points) {
        std::cerr << points.error() << '\n';
        return 1;
    }
    std::cout << *points << " points loaded into " << database << " as '" << dataset << "'\n";
}

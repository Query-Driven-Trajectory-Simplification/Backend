#include <iostream>
#include "backend/engine/io/parquet_loader.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <parquet_file>\n";
        return 1;
    }

    auto status = read_parquet(argv[1]);
    if (!status.ok()) {
        std::cerr << "Error: " << status.ToString() << "\n";
        return 1;
    }
}
#include "backend/engine/core/types.hpp"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/status.h>
#include <filesystem>
#include <parquet/arrow/reader.h>
#include <iostream>

namespace io {

// https://mojoauth.com/dev-guides/serialize-and-deserialize-parquet-with-cpp
// Think this one might be really good to look at. We should have defo looked at this one -
// when making the geolife reader for data_processing
arrow::Result<std::shared_ptr<arrow::Table>> read_parquet(const std::filesystem::path& path) {
    ARROW_ASSIGN_OR_RAISE(auto in, arrow::io::ReadableFile::Open(path.string()));

    parquet::ArrowReaderProperties arrow_props = parquet::default_arrow_reader_properties();
    arrow_props.set_use_threads(true);

    parquet::arrow::FileReaderBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Open(in));
    builder.memory_pool(arrow::default_memory_pool());
    builder.properties(arrow_props);

    ARROW_ASSIGN_OR_RAISE(auto reader, builder.Build());
    ARROW_ASSIGN_OR_RAISE(auto table, reader->ReadTable());

    std::cout << table->schema()->ToString() << "\n";
    std::cout << "row groups: " << reader->num_row_groups() << "\n";
    std::cout << "total rows: " << table->num_rows() << "\n";

    return table;
}

core::TrajectoryDb to_trajectory_db(const arrow::Table& table) {
    // walk the trajectory_id / t / lat / lon columns, fill the flat arrays,
    // push a new offset each time trajectory_id changes
}


void load_trajectories(const std::filesystem::path& path) {
    read_parquet(path);
}

}
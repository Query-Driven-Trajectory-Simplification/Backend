#include "backend/engine/core/types.hpp"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type_fwd.h>
#include <filesystem>
#include <memory>
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
    std::cout << "total rows: " << table->num_rows() << "\n";

    return table;
}

core::TrajectoryDb to_trajectory_db(const arrow::Table& table) {
    core::TrajectoryDb trajectoryDatabase{static_cast<size_t>(table.num_rows())};

    arrow::TableBatchReader batch_reader(table);
    std::shared_ptr<arrow::RecordBatch> batch;

    while (batch_reader.ReadNext(&batch).ok() && batch != nullptr) {
        auto id_col  = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName("trajectory_id"));
        auto t_col = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("t"));
        auto lat_col = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("lat"));
        auto lon_col = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("lon"));

        int64_t num_rows = batch->num_rows();
        for (int64_t i = 0; i < num_rows; ++i) {
            int32_t trajectory_id = id_col->Value(i);
            int64_t timestamp = t_col->Value(i);
            double lat = lat_col->Value(i);
            double lon = lon_col->Value(i);
        }
    }

    return trajectoryDatabase;
}


core::TrajectoryDb load_trajectories(const std::filesystem::path& path) {
    auto table = read_parquet(path);
    if (!table.ok()) {
        throw std::runtime_error(std::format("{}: {}", path.string(), table.status().ToString()));
    }
    return to_trajectory_db(**table);
}

}
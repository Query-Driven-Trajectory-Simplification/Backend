#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type_fwd.h>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <parquet/arrow/reader.h>
#include <stdexcept>
#include <vector>

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
    return reader->ReadTable();
}

core::TrajectoryDb to_trajectory_db(const arrow::Table& table) {
    const auto rows = static_cast<std::size_t>(table.num_rows());

    std::vector<int32_t> trajectory_ids;
    std::vector<int64_t> timestamps;
    std::vector<double> latitudes, longitudes;
    std::vector<std::uint64_t> offsets{0};
    timestamps.reserve(rows);
    latitudes.reserve(rows);
    longitudes.reserve(rows);

    arrow::TableBatchReader batch_reader(table);
    std::shared_ptr<arrow::RecordBatch> batch;
    arrow::Status status;

    while ((status = batch_reader.ReadNext(&batch)).ok() && batch != nullptr) {
        const auto id_col  = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName("trajectory_id"));
        const auto t_col = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("t"));
        const auto lat_col = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("lat"));
        const auto lon_col = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("lon"));

        for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
            const std::int32_t id = id_col->Value(i);
            if (trajectory_ids.empty() || id != trajectory_ids.back()) {
                if (!trajectory_ids.empty() && id < trajectory_ids.back())
                    throw std::runtime_error("trajectory_id not sorted; rows of a trip are split");
                if (!trajectory_ids.empty()) offsets.push_back(timestamps.size());
                trajectory_ids.push_back(id);
            }
            timestamps.push_back(t_col->Value(i));
            latitudes.push_back(lat_col->Value(i));
            longitudes.push_back(lon_col->Value(i));
        }
    }
    if (!status.ok())
        throw std::runtime_error(status.ToString());
    if (!trajectory_ids.empty()) offsets.push_back(timestamps.size());

    return core::TrajectoryDb(std::move(offsets), std::move(trajectory_ids), std::move(timestamps),
                              std::move(latitudes), std::move(longitudes));
}


core::TrajectoryDb load_trajectories(const std::filesystem::path& path) {
    auto table = read_parquet(path);
    if (!table.ok()) {
        throw std::runtime_error(std::format("{}: {}", path.string(), table.status().ToString()));
    }
    return to_trajectory_db(**table);
}

}
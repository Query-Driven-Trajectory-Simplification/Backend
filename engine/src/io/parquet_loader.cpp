#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <stdexcept>
#include <vector>

namespace io {

namespace {

arrow::Result<core::TrajectoryDb>
read_trajectory_db_from_batches(const std::filesystem::path &path) {
    ARROW_ASSIGN_OR_RAISE(auto in, arrow::io::ReadableFile::Open(path.string()));

    parquet::ArrowReaderProperties arrow_props = parquet::default_arrow_reader_properties();
    arrow_props.set_use_threads(true);
    arrow_props.set_pre_buffer(true);

    parquet::arrow::FileReaderBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Open(in));
    builder.memory_pool(arrow::default_memory_pool());
    builder.properties(arrow_props);

    ARROW_ASSIGN_OR_RAISE(auto reader, builder.Build());

    const auto rows = static_cast<std::size_t>(reader->parquet_reader()->metadata()->num_rows());

    std::vector<std::int32_t> trajectory_ids;
    std::vector<std::int64_t> timestamps;
    std::vector<double> latitudes, longitudes;
    std::vector<std::uint64_t> offsets{0};

    timestamps.reserve(rows);
    latitudes.reserve(rows);
    longitudes.reserve(rows);

    ARROW_ASSIGN_OR_RAISE(auto batch_reader, reader->GetRecordBatchReader());
    std::shared_ptr<arrow::RecordBatch> batch;

    while (true) {
        ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
        if (batch == nullptr)
            break;

        const auto id_col = std::dynamic_pointer_cast<arrow::Int32Array>(
            batch->GetColumnByName(std::string("trajectory_id")));
        const auto t_col =
            std::dynamic_pointer_cast<arrow::Int64Array>(batch->GetColumnByName(std::string("t")));
        const auto lat_col = std::dynamic_pointer_cast<arrow::DoubleArray>(
            batch->GetColumnByName(std::string("lat")));
        const auto lon_col = std::dynamic_pointer_cast<arrow::DoubleArray>(
            batch->GetColumnByName(std::string("lon")));

        if (!id_col || !t_col || !lat_col || !lon_col) {
            return arrow::Status::Invalid("Missing column or mismatched datatype for trajectory_id "
                                          "(int32), t (int64), lat (double), or lon (double)");
        }

        for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
            const std::int32_t id = id_col->Value(i);
            if (trajectory_ids.empty() || id != trajectory_ids.back()) {
                if (!trajectory_ids.empty() && id < trajectory_ids.back()) {
                    return arrow::Status::Invalid(
                        "trajectory_id not sorted; rows of a trip are split");
                }
                if (!trajectory_ids.empty())
                    offsets.push_back(timestamps.size());
                trajectory_ids.push_back(id);
            }
            timestamps.push_back(t_col->Value(i));
            latitudes.push_back(lat_col->Value(i));
            longitudes.push_back(lon_col->Value(i));
        }
    }

    if (!trajectory_ids.empty())
        offsets.push_back(timestamps.size());

    return core::TrajectoryDb(std::move(offsets), std::move(trajectory_ids), std::move(timestamps),
                              std::move(latitudes), std::move(longitudes));
}

} // namespace

core::TrajectoryDb load_trajectories(const std::filesystem::path &path) {
    auto db = read_trajectory_db_from_batches(path);
    if (!db.ok()) {
        throw std::runtime_error(std::format("{}: {}", path.string(), db.status().ToString()));
    }
    return std::move(*db);
}

} // namespace io
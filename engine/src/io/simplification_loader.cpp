#include "backend/engine/io/simplification_loader.hpp"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace io {

namespace {

using KeptById = std::unordered_map<std::int32_t, std::vector<std::size_t>>;

arrow::Result<KeptById> read_kept(const std::filesystem::path &path) {
    ARROW_ASSIGN_OR_RAISE(auto in, arrow::io::ReadableFile::Open(path.string()));
    parquet::arrow::FileReaderBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Open(in));
    ARROW_ASSIGN_OR_RAISE(auto reader, builder.Build());
    ARROW_ASSIGN_OR_RAISE(auto batch_reader, reader->GetRecordBatchReader());

    KeptById kept;
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
        ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
        if (batch == nullptr)
            break;
        const auto ids = std::dynamic_pointer_cast<arrow::Int32Array>(
            batch->GetColumnByName(std::string("trajectory_id")));
        const auto indices = std::dynamic_pointer_cast<arrow::Int64Array>(
            batch->GetColumnByName(std::string("point_index")));
        if (!ids || !indices)
            return arrow::Status::Invalid("Missing column or mismatched datatype for "
                                          "trajectory_id (int32) or point_index (int64)");
        if (ids->null_count() || indices->null_count())
            return arrow::Status::Invalid("Kept-point columns must not contain null values");
        for (std::int64_t row = 0; row < batch->num_rows(); ++row) {
            const auto index = indices->Value(row);
            if (index < 0)
                return arrow::Status::Invalid(std::format(
                    "negative point_index {} in trajectory {}", index, ids->Value(row)));
            kept[ids->Value(row)].push_back(static_cast<std::size_t>(index));
        }
    }
    return kept;
}

[[noreturn]] void fail(const std::filesystem::path &path, const std::string &message) {
    throw std::runtime_error(std::format("{}: {}", path.string(), message));
}

} // namespace

core::Simplification load_simplification(const std::filesystem::path &path,
                                         const core::TrajectoryDb &database) {
    auto read = read_kept(path);
    if (!read.ok())
        fail(path, read.status().ToString());
    KeptById kept = std::move(*read);

    core::Simplification simplification;
    for (std::size_t i = 0; i < database.size(); ++i) {
        const auto trajectory = database.trajectory(i);
        const auto found = kept.find(trajectory.trajectory_id);
        if (found == kept.end())
            fail(path, std::format("trajectory {} of the database has no kept points",
                                   trajectory.trajectory_id));

        auto &indices = found->second;
        std::ranges::sort(indices);
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        if (indices.back() >= trajectory.size())
            fail(path, std::format("point_index {} out of range in trajectory {} ({} points)",
                                   indices.back(), trajectory.trajectory_id, trajectory.size()));
        if (indices.front() != 0 || indices.back() != trajectory.size() - 1)
            fail(path, std::format("trajectory {} does not keep its first and last point",
                                   trajectory.trajectory_id));

        simplification.add_trajectory(indices);
        kept.erase(found);
    }
    if (!kept.empty())
        fail(path, std::format("trajectory {} is not in the database", kept.begin()->first));
    return simplification;
}

} // namespace io

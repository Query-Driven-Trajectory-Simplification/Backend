#include <arrow/status.h>
#include <filesystem>

namespace io {

arrow::Status read_parquet(const std::filesystem::path& path);

}
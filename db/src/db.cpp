#include "db/db.hpp"

#include <utility>

namespace db {

Result::Result(Result&& other) noexcept
    : result_(other.result_), owned_(std::exchange(other.owned_, false)) {}

Result::~Result() {
    if (owned_) duckdb_destroy_result(&result_);
}

std::int64_t Result::rows() const {
    return static_cast<std::int64_t>(duckdb_row_count(&result_));
}

std::int64_t Result::int64(std::size_t col, std::size_t row) const {
    return duckdb_value_int64(&result_, col, row);
}

double Result::number(std::size_t col, std::size_t row) const {
    return duckdb_value_double(&result_, col, row);
}

bool Result::is_null(std::size_t col, std::size_t row) const {
    return duckdb_value_is_null(&result_, col, row);
}

std::expected<Database, std::string> Database::open(const std::filesystem::path& path) {
    duckdb_database db = nullptr;
    char* error = nullptr;
    const char* file = path.empty() ? nullptr : path.c_str();
    if (duckdb_open_ext(file, &db, nullptr, &error) != DuckDBSuccess) {
        std::string message = error ? error : "could not open " + path.string();
        duckdb_free(error);
        return std::unexpected(message);
    }

    duckdb_connection con = nullptr;
    if (duckdb_connect(db, &con) != DuckDBSuccess) {
        duckdb_close(&db);
        return std::unexpected("could not connect to " + path.string());
    }
    return Database(db, con);
}

Database::Database(Database&& other) noexcept
    : db_(std::exchange(other.db_, nullptr)), con_(std::exchange(other.con_, nullptr)) {}

Database::~Database() {
    if (con_) duckdb_disconnect(&con_);
    if (db_) duckdb_close(&db_);
}

std::expected<Result, std::string> Database::query(
    const std::string& sql, std::initializer_list<std::string> params) {
    duckdb_result result;

    if (params.size() == 0) {
        if (duckdb_query(con_, sql.c_str(), &result) != DuckDBSuccess) {
            std::string message = duckdb_result_error(&result);
            duckdb_destroy_result(&result);
            return std::unexpected(message);
        }
        return Result(result);
    }

    duckdb_prepared_statement statement = nullptr;
    if (duckdb_prepare(con_, sql.c_str(), &statement) != DuckDBSuccess) {
        std::string message = duckdb_prepare_error(statement);
        duckdb_destroy_prepare(&statement);
        return std::unexpected(message);
    }

    idx_t index = 1;
    for (const auto& param : params) {
        duckdb_bind_varchar(statement, index++, param.c_str());
    }

    const bool ok = duckdb_execute_prepared(statement, &result) == DuckDBSuccess;
    duckdb_destroy_prepare(&statement);
    if (!ok) {
        std::string message = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        return std::unexpected(message);
    }
    return Result(result);
}

std::expected<void, std::string> create_schema(Database& db) {
    auto created = db.query(R"(
        CREATE TABLE IF NOT EXISTS points (
            trajectory_id    INTEGER   NOT NULL,
            dataset          VARCHAR   NOT NULL,  -- 'geolife', 'mlsimp', ...
            variant          VARCHAR   NOT NULL,  -- 'original', 'gaussian', 'model'
            compression_rate DOUBLE,              -- NULL for the original
            seq_no           INTEGER   NOT NULL,  -- 0-based position in the trajectory
            lat              DOUBLE    NOT NULL,
            lon              DOUBLE    NOT NULL,
            alt_m            DOUBLE,              -- NULL when unknown
            ts               TIMESTAMP NOT NULL   -- UTC
        );

        CREATE TABLE IF NOT EXISTS trajectories (
            dataset       VARCHAR NOT NULL,
            trajectory_id INTEGER NOT NULL,
            num_points    INTEGER NOT NULL,
            PRIMARY KEY (dataset, trajectory_id)
        );
    )");
    if (!created) return std::unexpected(created.error());
    return {};
}

namespace {

std::expected<std::int64_t, std::string> load_points_in_transaction(
    Database& db, const std::filesystem::path& parquet, const std::string& dataset) {
    auto deleted_points = db.query("DELETE FROM points WHERE dataset = $1", {dataset});
    if (!deleted_points) return std::unexpected(deleted_points.error());

    auto deleted_trajectories = db.query("DELETE FROM trajectories WHERE dataset = $1", {dataset});
    if (!deleted_trajectories) return std::unexpected(deleted_trajectories.error());

    // file_row_number keeps the file's point order: GeoLife has duplicate timestamps.
    // Sorting on insert keeps each trajectory's rows together on disk.
    auto inserted = db.query(R"(
        INSERT INTO points
        SELECT trajectory_id,
               $1,
               'original',
               NULL,
               (row_number() OVER (PARTITION BY trajectory_id ORDER BY file_row_number) - 1)::INTEGER,
               lat,
               lon,
               alt_m,
               make_timestamp(t * 1000000)
        FROM read_parquet($2, file_row_number = true)
        ORDER BY trajectory_id, file_row_number
    )", {dataset, parquet.string()});
    if (!inserted) return std::unexpected(inserted.error());

    auto summarized = db.query(R"(
        INSERT INTO trajectories
        SELECT dataset, trajectory_id, count(*)
        FROM points
        WHERE dataset = $1 AND variant = 'original'
        GROUP BY ALL
    )", {dataset});
    if (!summarized) return std::unexpected(summarized.error());

    return inserted->int64(0, 0);
}

}

std::expected<std::int64_t, std::string> load_points(
    Database& db, const std::filesystem::path& parquet, const std::string& dataset) {
    if (auto begun = db.query("BEGIN TRANSACTION"); !begun) return std::unexpected(begun.error());

    auto loaded = load_points_in_transaction(db, parquet, dataset);
    auto ended = db.query(loaded ? "COMMIT" : "ROLLBACK");
    if (!ended) return std::unexpected(ended.error());
    return loaded;
}

}

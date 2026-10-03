#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <initializer_list>
#include <string>

#include <duckdb.h>

namespace db
{

  // Query Result
  class Result
  {
  public:
    explicit Result(duckdb_result result) : result_(result) {}
    Result(Result &&other) noexcept;
    Result(const Result &) = delete;
    Result &operator=(const Result &) = delete;
    Result &operator=(Result &&) = delete;
    ~Result();

    std::int64_t rows() const;
    std::int64_t int64(std::size_t col, std::size_t row) const;
    double number(std::size_t col, std::size_t row) const;
    bool is_null(std::size_t col, std::size_t row) const;

  private:
    mutable duckdb_result result_;
    bool owned_ = true;
  };

  class Database
  {
  public:
    // Empty path open in-memory db.
    static std::expected<Database, std::string> open(const std::filesystem::path &path);

    Database(Database &&other) noexcept;
    Database(const Database &) = delete;
    Database &operator=(const Database &) = delete;
    Database &operator=(Database &&) = delete;
    ~Database();

    std::expected<Result, std::string> query(
        const std::string &sql, std::initializer_list<std::string> params = {});

  private:
    Database(duckdb_database db, duckdb_connection con) : db_(db), con_(con) {}

    duckdb_database db_ = nullptr;
    duckdb_connection con_ = nullptr;
  };

  // Creates points and trajectories tables.
  std::expected<void, std::string> create_schema(Database &db);

  std::expected<std::int64_t, std::string> load_points(
      Database &db, const std::filesystem::path &parquet, const std::string &dataset);

}
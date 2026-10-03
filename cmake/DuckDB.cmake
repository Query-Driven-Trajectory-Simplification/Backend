set(DUCKDB_VERSION 1.5.6)

if(APPLE)
    set(DUCKDB_ASSET osx-universal)
    set(DUCKDB_SHA256 e0bc007d9b0094c0970ac1847a8601d10aad07cbd2910ce586ec810f77b638d6)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(DUCKDB_ASSET linux-arm64)
    set(DUCKDB_SHA256 b72ed9f05003f5e9d2015f7ceada6416b377d9dd33169cdde3c9e33897856eee)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(DUCKDB_ASSET linux-amd64)
    set(DUCKDB_SHA256 b845005f5132a7d8180057c35e14a7626632258782f871a90861b19c1c03841b)
else()
    message(FATAL_ERROR "No prebuilt DuckDB for ${CMAKE_SYSTEM_NAME}")
endif()

FetchContent_Declare(duckdb
    URL https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/libduckdb-${DUCKDB_ASSET}.zip
    URL_HASH SHA256=${DUCKDB_SHA256})
FetchContent_MakeAvailable(duckdb)

add_library(duckdb::duckdb SHARED IMPORTED)
set_target_properties(duckdb::duckdb PROPERTIES
    IMPORTED_LOCATION ${duckdb_SOURCE_DIR}/libduckdb${CMAKE_SHARED_LIBRARY_SUFFIX}
    INTERFACE_INCLUDE_DIRECTORIES ${duckdb_SOURCE_DIR})

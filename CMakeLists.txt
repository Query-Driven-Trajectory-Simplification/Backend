cmake_minimum_required(VERSION 4.0)
project(backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)
# Arrow's ARROW_ASSIGN_OR_RAISE uses __COUNTER__, which clang -Wpedantic flags
add_compile_options($<$<CXX_COMPILER_ID:Clang>:-Wno-c2y-extensions>)

find_package(Arrow REQUIRED QUIET)
find_package(Parquet REQUIRED QUIET)

enable_testing()

add_subdirectory(data_processing)
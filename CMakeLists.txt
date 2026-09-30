cmake_minimum_required(VERSION 4.0)
project(Backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd / IDE tooling


add_subdirectory(
    src/
)

target_include_directories(Backend PRIVATE include)
target_compile_options(Backend PRIVATE -Wall -Wextra)
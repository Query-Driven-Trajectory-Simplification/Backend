add_library(engine
    src/io/parquet_loader.cpp
)

target_include_directories(engine PUBLIC include)
target_link_libraries(engine PUBLIC Arrow::arrow_shared Parquet::parquet_shared)

add_executable(run
    apps/run.cpp
)

target_link_libraries(run PRIVATE engine)
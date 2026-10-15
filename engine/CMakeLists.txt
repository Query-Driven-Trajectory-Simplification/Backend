add_library(engine
    src/core/trajectory_db.cpp
    src/io/parquet_loader.cpp
    src/pipeline/pipeline.cpp
)

target_include_directories(engine PUBLIC include)
target_link_libraries(engine PRIVATE Arrow::arrow_shared Parquet::parquet_shared)

add_executable(run
    apps/run.cpp
)

target_link_libraries(run PRIVATE engine)
add_executable(run
    apps/run.cpp
)
target_link_libraries(run PRIVATE Arrow::arrow_shared Parquet::parquet_shared)
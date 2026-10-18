add_library(engine
    src/core/trajectory_db.cpp
    src/io/frontend_export.cpp
    src/io/parquet_loader.cpp
    src/pipeline/pipeline.cpp
    src/query/box_query.cpp
    src/query/range_query.cpp
)

target_include_directories(engine PUBLIC include)
target_link_libraries(engine PRIVATE Arrow::arrow_shared Parquet::parquet_shared)

add_executable(run
    apps/run.cpp
)

target_link_libraries(run PRIVATE engine)

# googletest is fetched in data_processing/CMakeLists.txt
add_executable(engine_tests
    tests/box_query_test.cpp
    tests/range_query_test.cpp
)
target_link_libraries(engine_tests PRIVATE engine GTest::gtest_main)
gtest_discover_tests(engine_tests)

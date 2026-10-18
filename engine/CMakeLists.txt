add_library(engine
    src/core/trajectory_db.cpp
    src/io/parquet_loader.cpp
    src/pipeline/pipeline.cpp
    src/selection/rdp.cpp
    src/selection/sed.cpp
    src/selection/selector.cpp
)

target_include_directories(engine PUBLIC include)
target_link_libraries(engine PRIVATE Arrow::arrow_shared Parquet::parquet_shared)

add_executable(run
    apps/run.cpp
)

target_link_libraries(run PRIVATE engine)

add_executable(engine_tests
    tests/projection_test.cpp
    tests/rdp_test.cpp
    tests/sed_test.cpp
)
target_link_libraries(engine_tests PRIVATE engine GTest::gtest_main)
gtest_discover_tests(engine_tests)
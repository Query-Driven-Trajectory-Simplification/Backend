add_library(engine
    src/core/trajectory_db.cpp
    src/io/frontend_export.cpp
    src/io/parquet_loader.cpp
    src/pipeline/pipeline.cpp
    src/query/box_query.cpp
    src/query/range_query.cpp
    src/selection/rdp.cpp
    src/selection/selector.cpp
    src/split/split.cpp
    src/allocation/budget_allocator.cpp
    src/allocation/size_proportional.cpp
)

target_include_directories(engine PUBLIC include)
target_link_libraries(engine PRIVATE Arrow::arrow_shared Parquet::parquet_shared)

add_executable(run
    apps/run.cpp
)

target_link_libraries(run PRIVATE engine)

add_executable(engine_tests
    tests/projection_test.cpp
    tests/selectors/rdp_test.cpp
    tests/split/time_split_test.cpp
    tests/split/random_split_test.cpp
    tests/rdp_test.cpp
    tests/size_proportional_test.cpp
)
target_link_libraries(engine_tests PRIVATE engine GTest::gtest_main)
gtest_discover_tests(engine_tests)

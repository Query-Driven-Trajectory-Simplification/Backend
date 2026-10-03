add_library(db
    src/db.cpp
)
target_include_directories(db PUBLIC include)
target_link_libraries(db PUBLIC duckdb::duckdb)

add_executable(load_db
    apps/load_db.cpp
)
target_link_libraries(load_db PRIVATE db)

add_executable(db_tests
    tests/db_test.cpp
)
target_link_libraries(db_tests PRIVATE db GTest::gtest_main)
gtest_discover_tests(db_tests)

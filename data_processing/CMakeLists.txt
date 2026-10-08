add_library(data_processing
    src/readers/geolife.cpp
    src/filters/boundingbox.cpp
    src/filters/filters.cpp
    src/filters/minsize.cpp
    src/filters/timegap.cpp
)
target_include_directories(data_processing PUBLIC include)
target_link_libraries(data_processing PUBLIC Arrow::arrow_shared Parquet::parquet_shared)

add_executable(convert
    apps/convert.cpp
)
target_link_libraries(convert PRIVATE data_processing)

include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.zip
    SYSTEM)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(data_processing_tests
    tests/geolife_test.cpp
)
target_link_libraries(data_processing_tests PRIVATE data_processing GTest::gtest_main)
gtest_discover_tests(data_processing_tests)

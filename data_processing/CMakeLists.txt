add_library(data_processing
    src/readers/geolife.cpp
    src/filters/boundingbox.cpp
    src/filters/distance.cpp
    src/filters/filters.cpp
    src/filters/mintrip.cpp
    src/filters/staypoint.cpp
    src/filters/timegap.cpp
)
target_include_directories(data_processing PUBLIC include)
target_link_libraries(data_processing PUBLIC Arrow::arrow_shared Parquet::parquet_shared)

add_executable(convert
    apps/convert.cpp
)
target_link_libraries(convert PRIVATE data_processing)
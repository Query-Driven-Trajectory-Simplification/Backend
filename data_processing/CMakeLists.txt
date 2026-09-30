add_library(data_processing
    src/plt_reader.cpp
)
target_include_directories(data_processing PUBLIC include)

add_executable(convert_plt
    apps/convert_plt.cpp
)
target_link_libraries(convert_plt PRIVATE data_processing)

add_executable(teacher_harness
    harness.cpp
    caches.cpp
    baselines.cpp
    objective.cpp
    curves.cpp
    teacher.cpp
)

target_link_libraries(teacher_harness PRIVATE engine)

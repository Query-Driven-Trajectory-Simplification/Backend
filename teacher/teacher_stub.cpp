// Returns no arms, so only the baselines run. Replace with teacher.cpp
#include "teacher.hpp"

namespace teacher {

TeacherOutput run_teacher(const core::TrajectoryDb &, const TeacherCaches &, std::uint64_t,
                          std::span<const double>) {
    return {};
}

} // namespace teacher

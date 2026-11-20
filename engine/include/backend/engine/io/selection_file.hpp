#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/range_query.hpp"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace io {

void export_database(const std::filesystem::path &dir, const core::TrajectoryDb &database);

void export_budgets(const std::filesystem::path &dir, std::string_view name,
                    std::span<const std::size_t> budgets);

std::size_t export_range_hits(const std::filesystem::path &path, const core::TrajectoryDb &database,
                              std::span<const query::RangeQuery> queries);

core::KeptSets read_selection(const std::filesystem::path &path,
                              const core::TrajectoryDb &database);

} // namespace io

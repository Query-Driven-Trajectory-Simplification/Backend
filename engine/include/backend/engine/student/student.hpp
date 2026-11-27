#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/student/student_model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace student {

struct SimplifyResult {
    core::KeptSets kept;
    double density_ms = 0.0;
    double feature_ms = 0.0;
    double ml_ms = 0.0;
    double pool_ms = 0.0;
    double decode_ms = 0.0;
};

// Full pure-C++ end-to-end Student inference (zero Python, zero Frechet).
SimplifyResult simplify_database(const core::TrajectoryDb &db, std::size_t total_budget,
                                 const StudentModel &model, int rb_passes = 2,
                                 bool rdp_only = false);

// Out-of-fold pure-C++ inference across fold-isolated models (dual-price aligned at 1% ref share).
SimplifyResult simplify_database_oof(const core::TrajectoryDb &db, std::size_t total_budget,
                                     std::span<const StudentModel> fold_models,
                                     std::span<const std::uint32_t> fold_ids, int rb_passes = 2,
                                     bool rdp_only = false);

} // namespace student
